// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::Int32Builder;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::datasource::datasource::{TableProvider, TableType};
use datafusion::datasource::provider_as_source;
use datafusion::error::Result;
use datafusion::execution::context::{
    default_session_builder, SessionState, TaskContext,
};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::common::SizedRecordBatchStream;
use datafusion::physical_plan::expressions::PhysicalSortExpr;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MemTrackingMetrics};
use datafusion::physical_plan::planner::DefaultPhysicalPlanner;
use datafusion::physical_plan::{
    displayable, DisplayFormatType, ExecutionPlan, Partitioning, PhysicalPlanner,
    SendableRecordBatchStream, Statistics,
};
use datafusion::prelude::*;
use datafusion_common::DataFusionError;
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::logical_plan::AggWithGrouping;
use datafusion_expr::{
    aggregate_function, LogicalPlan, LogicalPlanBuilder,
    TableProviderAggregationPushDown, UNNAMED_TABLE,
};
use datafusion_optimizer::optimizer::Optimizer;
use datafusion_optimizer::{OptimizerContext, OptimizerRule};
use std::ops::Deref;
use std::sync::Arc;

fn create_batch(value: i32, num_rows: usize) -> Result<RecordBatch> {
    let mut builder1 = Int32Builder::with_capacity(num_rows);
    let mut builder2 = Int32Builder::with_capacity(num_rows);
    for _ in 0..num_rows {
        builder1.append_value(value);
        builder2.append_value(value);
    }

    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("flag", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ])),
        vec![Arc::new(builder1.finish()), Arc::new(builder2.finish())],
    )?)
}

#[derive(Debug)]
struct CustomAggregationPlan {
    schema: SchemaRef,
    pushed_aggs: Vec<CustomAggregateFunction>,
    one_batch: RecordBatch,
}

impl ExecutionPlan for CustomAggregationPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(4)
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        unreachable!()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let metrics = ExecutionPlanMetricsSet::new();
        let tracking_metrics =
            MemTrackingMetrics::new(&metrics, context.memory_pool(), partition);
        Ok(Box::pin(SizedRecordBatchStream::new(
            self.schema(),
            vec![Arc::new(self.one_batch.clone())],
            tracking_metrics,
        )))
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    "CustomAggregationPlan: pushed_aggs={:?}",
                    self.pushed_aggs
                )
            }
        }
    }

    fn statistics(&self) -> Statistics {
        // here we could provide more accurate statistics
        // but we want to test the filter pushdown not the CBOs
        Statistics::default()
    }
}

#[derive(Debug, Clone)]
pub enum CustomAggregateFunction {
    Count(Column),
}

#[derive(Clone)]
pub struct CustomAggregationProvider {
    zero_batch: RecordBatch,
    one_batch: RecordBatch,
}

#[async_trait]
impl TableProvider for CustomAggregationProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.zero_batch.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &SessionState,
        _proj: Option<&Vec<usize>>,
        _filters: &[Expr],
        agg_with_grouping: Option<&AggWithGrouping>,
        _: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(AggWithGrouping {
            group_expr: _,
            agg_expr,
            schema,
        }) = agg_with_grouping
        {
            let aggs = agg_expr
                .iter()
                .map(|e| match e {
                    Expr::AggregateFunction(agg) => Ok(agg),
                    _ => Err(DataFusionError::Plan(
                        "Invalid plan, pushed aggregate functions contains unsupported"
                            .to_string(),
                    )),
                })
                .collect::<Result<Vec<_>>>()?;

            let pushed_aggs = aggs
                .into_iter()
                .map(|agg| {
                    let AggregateFunction {
                        fun,
                        args,
                        ..
                    } = agg;

                    args
                        .iter()
                        .map(|expr| {
                            // 下推的聚合函数的参数必须是column列
                            match expr {
                                Expr::Column(c) => Ok(c),
                                _ => Err(DataFusionError::Internal(format!("Pushed aggregate functions's args contains non-column: {expr:?}.")))
                            }
                        })
                        .collect::<Result<Vec<_>>>()
                        .and_then(|columns| {
                            // 将下推的聚合函数转换为中间结构
                            match fun {
                                aggregate_function::AggregateFunction::Count => {
                                    let column = columns
                                        .first()
                                        .ok_or_else(|| DataFusionError::Internal("Pushed aggregate functions's args is none.".to_string()))?
                                        .deref()
                                        .clone();
                                    Ok(CustomAggregateFunction::Count(column))
                                },
                                // aggregate_function::AggregateFunction::Max => {},
                                // aggregate_function::AggregateFunction::Min => {},
                                _ => Err(DataFusionError::Internal("Pushed aggregate functions's args is none.".to_string())),
                            }
                        })
                })
                .collect::<Result<Vec<_>>>()?;

            return Ok(Arc::new(CustomAggregationPlan {
                schema: Arc::new(schema.as_ref().into()),
                pushed_aggs,
                one_batch: self.one_batch.clone(),
            }));
        }

        Ok(Arc::new(CustomAggregationPlan {
            schema: self.schema(),
            pushed_aggs: vec![],
            one_batch: self.one_batch.clone(),
        }))
    }

    fn supports_aggregate_pushdown(
        &self,
        group_expr: &[Expr],
        aggr_expr: &[Expr],
    ) -> Result<TableProviderAggregationPushDown> {
        if !group_expr.is_empty() {
            return Ok(TableProviderAggregationPushDown::Unsupported);
        }

        let result = if aggr_expr.iter().all(|e| {
            match e {
                Expr::AggregateFunction(AggregateFunction {
                    fun,
                    args,
                    distinct,
                    filter,
                }) => {
                    let support_agg_func = match fun {
                        aggregate_function::AggregateFunction::Count => true,
                        // aggregate_function::AggregateFunction::Max => {},
                        // aggregate_function::AggregateFunction::Min => {},
                        _ => false,
                    };

                    support_agg_func
                        && args.len() == 1
                        && matches!(args[0], Expr::Column(_))
                        && !distinct
                        && filter.is_none()
                }
                _ => false,
            }
        }) {
            TableProviderAggregationPushDown::Ungrouped
        } else {
            TableProviderAggregationPushDown::Unsupported
        };

        Ok(result)
    }
}

fn observe(_plan: &LogicalPlan, _rule: &dyn OptimizerRule) {}

fn optimize_plan(plan: &LogicalPlan) -> Result<LogicalPlan> {
    let opt = Optimizer::new();
    let config = OptimizerContext::new().with_skip_failing_rules(false);

    opt.optimize(plan, &config, &observe)
}

fn test_table_scan() -> Result<LogicalPlan> {
    let provider = Arc::new(CustomAggregationProvider {
        zero_batch: create_batch(0, 10)?,
        one_batch: create_batch(1, 5)?,
    });

    LogicalPlanBuilder::scan(UNNAMED_TABLE, provider_as_source(provider), None)?.build()
}

#[tokio::test]
async fn test_count_with_group() -> Result<()> {
    let plan = LogicalPlanBuilder::from(test_table_scan()?)
        .aggregate(vec![col("flag")], vec![count(col("value"))])?
        .build()?;

    let opt_plan = optimize_plan(&plan)?;

    let result_str = format!("{opt_plan:?}");

    assert_eq!(
        "\
    Aggregate: groupBy=[[?table?.flag]], aggr=[[COUNT(?table?.value)]]\
    \n  TableScan: ?table? projection=[flag, value]",
        result_str
    );

    let planner = DefaultPhysicalPlanner::default();
    let optimized_physical_plan = planner
        .create_physical_plan(
            &opt_plan,
            &default_session_builder(SessionConfig::default()),
        )
        .await?;

    let result_str =
        format!("{}", displayable(optimized_physical_plan.as_ref()).indent());

    assert_eq!(
        "\
        AggregateExec: mode=FinalPartitioned, gby=[flag@0 as flag], aggr=[COUNT(?table?.value)]\
        \n  CoalesceBatchesExec: target_batch_size=8192\
        \n    RepartitionExec: partitioning=Hash([Column { name: \"flag\", index: 0 }], 8), input_partitions=8\
        \n      AggregateExec: mode=Partial, gby=[flag@0 as flag], aggr=[COUNT(?table?.value)]\
        \n        RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=4\
        \n          CustomAggregationPlan: pushed_aggs=[]\
        \n",
        result_str
    );

    Ok(())
}

#[tokio::test]
async fn test_count_without_group() -> Result<()> {
    let plan = LogicalPlanBuilder::from(test_table_scan()?)
        .aggregate(Vec::<Expr>::new(), vec![count(col("value"))])?
        .build()?;

    let opt_plan = optimize_plan(&plan)?;

    let result_str = format!("{opt_plan:?}");

    assert_eq!(
        "\
    Projection: SUM(COUNT(?table?.value)) AS COUNT(?table?.value)\
    \n  Aggregate: groupBy=[[]], aggr=[[SUM(COUNT(?table?.value))]]\
    \n    TableScan: ?table?, grouping=[], agg=[COUNT(?table?.value)]",
        result_str
    );

    let planner = DefaultPhysicalPlanner::default();
    let optimized_physical_plan = planner
        .create_physical_plan(
            &opt_plan,
            &default_session_builder(SessionConfig::default()),
        )
        .await?;

    let result_str =
        format!("{}", displayable(optimized_physical_plan.as_ref()).indent());

    assert_eq!(
        "\
        ProjectionExec: expr=[SUM(COUNT(?table?.value))@0 as COUNT(?table?.value)]\
        \n  AggregateExec: mode=Final, gby=[], aggr=[SUM(COUNT(?table?.value))]\
        \n    CoalescePartitionsExec\
        \n      AggregateExec: mode=Partial, gby=[], aggr=[SUM(COUNT(?table?.value))]\
        \n        RepartitionExec: partitioning=RoundRobinBatch(8), input_partitions=4\
        \n          CustomAggregationPlan: pushed_aggs=[Count(Column { relation: Some(\"?table?\"), name: \"value\" })]\
        \n",
        result_str
    );

    Ok(())
}

#[tokio::test]
async fn test_max_with_group() -> Result<()> {
    let plan = LogicalPlanBuilder::from(test_table_scan()?)
        .aggregate(vec![col("value")], vec![max(col("value"))])?
        .build()?;

    let opt_plan = optimize_plan(&plan)?;

    let result_str = format!("{opt_plan:?}");

    assert_eq!(
        "\
    Aggregate: groupBy=[[?table?.value]], aggr=[[MAX(?table?.value)]]\
    \n  TableScan: ?table? projection=[value]",
        result_str
    );

    Ok(())
}

#[tokio::test]
async fn test_max_without_group() -> Result<()> {
    let plan = LogicalPlanBuilder::from(test_table_scan()?)
        .aggregate(Vec::<Expr>::new(), vec![max(col("value"))])?
        .build()?;

    let opt_plan = optimize_plan(&plan)?;

    let result_str = format!("{opt_plan:?}");

    assert_eq!(
        "\
    Aggregate: groupBy=[[]], aggr=[[MAX(?table?.value)]]\
    \n  TableScan: ?table? projection=[value]",
        result_str
    );

    Ok(())
}

#[tokio::test]
async fn test_min_with_group() -> Result<()> {
    let plan = LogicalPlanBuilder::from(test_table_scan()?)
        .aggregate(vec![col("value")], vec![min(col("value"))])?
        .build()?;

    let opt_plan = optimize_plan(&plan)?;

    let result_str = format!("{opt_plan:?}");

    assert_eq!(
        "\
    Aggregate: groupBy=[[?table?.value]], aggr=[[MIN(?table?.value)]]\
    \n  TableScan: ?table? projection=[value]",
        result_str
    );

    Ok(())
}

#[tokio::test]
async fn test_min_without_group() -> Result<()> {
    let plan = LogicalPlanBuilder::from(test_table_scan()?)
        .aggregate(Vec::<Expr>::new(), vec![min(col("value"))])?
        .build()?;

    let opt_plan = optimize_plan(&plan)?;

    let result_str = format!("{opt_plan:?}");

    assert_eq!(
        "\
    Aggregate: groupBy=[[]], aggr=[[MIN(?table?.value)]]\
    \n  TableScan: ?table? projection=[value]",
        result_str
    );

    Ok(())
}

#[tokio::test]
async fn test_sum_with_group() -> Result<()> {
    let plan = LogicalPlanBuilder::from(test_table_scan()?)
        .aggregate(vec![col("value")], vec![sum(col("value"))])?
        .build()?;

    let opt_plan = optimize_plan(&plan)?;

    let result_str = format!("{opt_plan:?}");

    assert_eq!(
        "\
    Aggregate: groupBy=[[?table?.value]], aggr=[[SUM(?table?.value)]]\
    \n  TableScan: ?table? projection=[value]",
        result_str
    );

    Ok(())
}

#[tokio::test]
async fn test_sum_without_group() -> Result<()> {
    let plan = LogicalPlanBuilder::from(test_table_scan()?)
        .aggregate(Vec::<Expr>::new(), vec![sum(col("value"))])?
        .build()?;

    let opt_plan = optimize_plan(&plan)?;

    let result_str = format!("{opt_plan:?}");

    assert_eq!(
        "\
    Aggregate: groupBy=[[]], aggr=[[SUM(?table?.value)]]\
    \n  TableScan: ?table? projection=[value]",
        result_str
    );

    Ok(())
}
