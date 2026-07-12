import * as path from "node:path";

import * as apigateway from "aws-cdk-lib/aws-apigateway";
import * as athena from "aws-cdk-lib/aws-athena";
import * as cloudwatch from "aws-cdk-lib/aws-cloudwatch";
import * as cdk from "aws-cdk-lib";
import * as events from "aws-cdk-lib/aws-events";
import * as eventTargets from "aws-cdk-lib/aws-events-targets";
import * as glue from "aws-cdk-lib/aws-glue";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as lambdaEventSources from "aws-cdk-lib/aws-lambda-event-sources";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as sqs from "aws-cdk-lib/aws-sqs";
import { Construct } from "constructs";

const VALIDATED_COLUMNS: glue.CfnTable.ColumnProperty[] = [
  { name: "schema_version", type: "int" },
  { name: "event_id", type: "string" },
  { name: "run_id", type: "string" },
  { name: "attempt_id", type: "string" },
  { name: "sequence", type: "bigint" },
  { name: "event_type", type: "string" },
  { name: "event_time", type: "string" },
  { name: "ingested_at", type: "string" },
  { name: "stage", type: "string" },
  { name: "span", type: "string" },
  { name: "status", type: "string" },
  { name: "elapsed_seconds", type: "double" },
  { name: "processed_frames", type: "bigint" },
  { name: "total_frames", type: "bigint" },
  { name: "progress_percent", type: "double" },
  { name: "eta_seconds", type: "double" },
  { name: "clip_duration_seconds", type: "double" },
  { name: "clip_frame_count", type: "bigint" },
  { name: "clip_width", type: "bigint" },
  { name: "clip_height", type: "bigint" },
  { name: "clip_fps", type: "double" },
  { name: "clip_size_bytes", type: "bigint" },
  { name: "duration_bucket", type: "string" },
  { name: "resolution_bucket", type: "string" },
  { name: "environment", type: "string" },
  { name: "service", type: "string" },
  { name: "execution_environment", type: "string" },
  { name: "runpod_endpoint_id", type: "string" },
  { name: "attempt_number", type: "bigint" },
  { name: "processor_version", type: "string" },
  { name: "backend", type: "string" },
  { name: "model", type: "string" },
  { name: "gpu_type", type: "string" },
  { name: "cold_start", type: "boolean" },
  { name: "cache_hit", type: "boolean" },
  { name: "model_build_seconds", type: "double" },
  { name: "predictor_lock_wait_seconds", type: "double" },
  { name: "data_ready_seconds", type: "double" },
  { name: "presentation_tail_seconds", type: "double" },
  { name: "rss_mb", type: "double" },
  { name: "peak_rss_mb", type: "double" },
  { name: "cuda_allocated_mb", type: "double" },
  { name: "cuda_reserved_mb", type: "double" },
  { name: "cuda_peak_mb", type: "double" },
  { name: "gpu_utilization_pct", type: "double" },
  { name: "error_class", type: "string" },
  { name: "error_code", type: "string" },
  { name: "artifact_type", type: "string" },
  { name: "artifact_size_bytes", type: "bigint" },
  { name: "milliseconds_per_frame", type: "double" },
  { name: "timing_basis", type: "string" },
  { name: "measurements_json", type: "string" },
  { name: "attributes_json", type: "string" },
];

const DAILY_AGGREGATE_COLUMNS: glue.CfnTable.ColumnProperty[] = [
  { name: "event_type", type: "string" },
  { name: "stage", type: "string" },
  { name: "span", type: "string" },
  { name: "backend", type: "string" },
  { name: "gpu_type", type: "string" },
  { name: "cold_start", type: "boolean" },
  { name: "duration_bucket", type: "string" },
  { name: "resolution_bucket", type: "string" },
  { name: "samples", type: "bigint" },
  { name: "p50_seconds", type: "double" },
  { name: "p90_seconds", type: "double" },
  { name: "p95_seconds", type: "double" },
  { name: "average_seconds", type: "double" },
  { name: "failures", type: "bigint" },
];

export class ProcessingAnalyticsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const lambdaSource = path.join(__dirname, "..", "lambda_src");
    const dataBucket = new s3.Bucket(this, "EventLake", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: false,
      removalPolicy: cdk.RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE,
      lifecycleRules: [
        { id: "expire-raw-events", prefix: "raw/", expiration: cdk.Duration.days(90) },
        {
          id: "expire-validated-events",
          prefix: "validated/",
          expiration: cdk.Duration.days(365),
        },
        {
          id: "expire-validated-progress",
          prefix: "validated/",
          tagFilters: { "event-class": "progress" },
          expiration: cdk.Duration.days(90),
        },
        {
          id: "expire-athena-results",
          prefix: "athena-results/",
          expiration: cdk.Duration.days(30),
        },
        {
          id: "expire-dashboard-results",
          prefix: "dashboard-results/",
          expiration: cdk.Duration.days(7),
        },
      ],
    });

    const deadLetterQueue = new sqs.Queue(this, "TelemetryDeadLetterQueue", {
      fifo: true,
      contentBasedDeduplication: false,
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      enforceSSL: true,
      retentionPeriod: cdk.Duration.days(14),
    });
    const telemetryQueue = new sqs.Queue(this, "TelemetryQueue", {
      fifo: true,
      contentBasedDeduplication: false,
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      enforceSSL: true,
      retentionPeriod: cdk.Duration.days(4),
      visibilityTimeout: cdk.Duration.seconds(210),
      deadLetterQueue: { queue: deadLetterQueue, maxReceiveCount: 5 },
    });
    deadLetterQueue.applyRemovalPolicy(cdk.RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE);
    telemetryQueue.applyRemovalPolicy(cdk.RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE);

    const ingestSecret = new secretsmanager.Secret(this, "IngestHmacSecret", {
      description: "HMAC secret used only by the Cloudflare Worker analytics exporter",
      generateSecretString: { excludePunctuation: true, passwordLength: 64 },
      removalPolicy: cdk.RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE,
    });
    const dashboardSecret = new secretsmanager.Secret(this, "DashboardHmacSecret", {
      description: "HMAC secret used only by the private dashboard Cloudflare Worker",
      generateSecretString: { excludePunctuation: true, passwordLength: 64 },
      removalPolicy: cdk.RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE,
    });

    const ingestLogGroup = new logs.LogGroup(this, "IngestFunctionLogs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const ingestFunction = new lambda.Function(this, "IngestFunction", {
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      code: lambda.Code.fromAsset(lambdaSource),
      handler: "ingest.handler",
      timeout: cdk.Duration.seconds(10),
      memorySize: 256,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: ingestLogGroup,
      environment: {
        TELEMETRY_QUEUE_URL: telemetryQueue.queueUrl,
        INGEST_SECRET_ARN: ingestSecret.secretArn,
        MAX_CLOCK_SKEW_SECONDS: "300",
        MAX_EVENT_BYTES: "65536",
      },
    });
    ingestSecret.grantRead(ingestFunction);
    telemetryQueue.grantSendMessages(ingestFunction);

    const consumerLogGroup = new logs.LogGroup(this, "ConsumerFunctionLogs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const consumerFunction = new lambda.Function(this, "ConsumerFunction", {
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      code: lambda.Code.fromAsset(lambdaSource),
      handler: "consumer.handler",
      timeout: cdk.Duration.seconds(30),
      memorySize: 512,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: consumerLogGroup,
      environment: {
        EVENT_BUCKET: dataBucket.bucketName,
      },
    });
    consumerFunction.addEventSource(
      new lambdaEventSources.SqsEventSource(telemetryQueue, {
        batchSize: 10,
        reportBatchItemFailures: true,
      }),
    );
    consumerFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:PutObject", "s3:PutObjectTagging"],
        resources: [dataBucket.arnForObjects("raw/*"), dataBucket.arnForObjects("validated/*")],
      }),
    );

    const apiAccessLogs = new logs.LogGroup(this, "IngestApiAccessLogs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const ingestApi = new apigateway.RestApi(this, "IngestApi", {
      description: "Private HMAC-authenticated processing telemetry ingress",
      cloudWatchRole: true,
      cloudWatchRoleRemovalPolicy: cdk.RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE,
      deployOptions: {
        stageName: "v1",
        accessLogDestination: new apigateway.LogGroupLogDestination(apiAccessLogs),
        accessLogFormat: apigateway.AccessLogFormat.jsonWithStandardFields({
          caller: false,
          httpMethod: true,
          ip: true,
          protocol: true,
          requestTime: true,
          resourcePath: true,
          responseLength: true,
          status: true,
          user: false,
        }),
        dataTraceEnabled: false,
        loggingLevel: apigateway.MethodLoggingLevel.ERROR,
        metricsEnabled: true,
        throttlingBurstLimit: 100,
        throttlingRateLimit: 50,
        tracingEnabled: true,
      },
      endpointTypes: [apigateway.EndpointType.REGIONAL],
    });
    ingestApi.root
      .addResource("events")
      .addMethod("POST", new apigateway.LambdaIntegration(ingestFunction, { proxy: true }));

    const analyticsMetricNamespace = "WhoDoIRunLike/Analytics";
    new logs.MetricFilter(this, "IngestRejectedMetricFilter", {
      logGroup: ingestLogGroup,
      metricNamespace: analyticsMetricNamespace,
      metricName: "IngestRejected",
      metricValue: "1",
      filterPattern: logs.FilterPattern.any(
        logs.FilterPattern.stringValue("$.message", "=", "telemetry authentication rejected"),
        logs.FilterPattern.stringValue("$.message", "=", "telemetry event rejected"),
      ),
    });
    new logs.MetricFilter(this, "IngestUnavailableMetricFilter", {
      logGroup: ingestLogGroup,
      metricNamespace: analyticsMetricNamespace,
      metricName: "IngestUnavailable",
      metricValue: "1",
      filterPattern: logs.FilterPattern.stringValue(
        "$.message",
        "=",
        "telemetry ingestion failed",
      ),
    });
    new logs.MetricFilter(this, "EventLakeWriteFailureMetricFilter", {
      logGroup: consumerLogGroup,
      metricNamespace: analyticsMetricNamespace,
      metricName: "EventLakeWriteFailure",
      metricValue: "1",
      filterPattern: logs.FilterPattern.stringValue(
        "$.message",
        "=",
        "telemetry event-lake write failed",
      ),
    });

    const databaseName = "whodoirunlike_analytics";
    const tableName = "processing_events";
    const database = new glue.CfnDatabase(this, "AnalyticsDatabase", {
      catalogId: this.account,
      databaseInput: { name: databaseName, description: "Processing Attempt telemetry" },
    });
    const processingEvents = new glue.CfnTable(this, "ProcessingEventsTable", {
      catalogId: this.account,
      databaseName,
      tableInput: {
        name: tableName,
        description: "Validated and flattened processing telemetry events",
        tableType: "EXTERNAL_TABLE",
        parameters: {
          classification: "json",
          "projection.enabled": "true",
          "projection.event_date.type": "date",
          "projection.event_date.range": "2026-01-01,NOW",
          "projection.event_date.format": "yyyy-MM-dd",
          "projection.event_hour.type": "integer",
          "projection.event_hour.range": "0,23",
          "projection.event_hour.digits": "2",
          "storage.location.template": `s3://${dataBucket.bucketName}/validated/event_date=\${event_date}/event_hour=\${event_hour}/`,
        },
        partitionKeys: [
          { name: "event_date", type: "string" },
          { name: "event_hour", type: "string" },
        ],
        storageDescriptor: {
          columns: VALIDATED_COLUMNS,
          compressed: true,
          inputFormat: "org.apache.hadoop.mapred.TextInputFormat",
          outputFormat: "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
          location: `s3://${dataBucket.bucketName}/validated/`,
          serdeInfo: {
            serializationLibrary: "org.openx.data.jsonserde.JsonSerDe",
            parameters: { "ignore.malformed.json": "false" },
          },
        },
      },
    });
    processingEvents.addDependency(database);

    const aggregateTableName = "daily_stage_performance";
    const dailyStagePerformance = new glue.CfnTable(this, "DailyStagePerformanceTable", {
      catalogId: this.account,
      databaseName,
      tableInput: {
        name: aggregateTableName,
        description: "Indefinitely retained daily stage/span Parquet aggregates",
        tableType: "EXTERNAL_TABLE",
        parameters: {
          classification: "parquet",
          "projection.enabled": "true",
          "projection.event_date.type": "date",
          "projection.event_date.range": "2026-01-01,NOW",
          "projection.event_date.format": "yyyy-MM-dd",
          "storage.location.template": `s3://${dataBucket.bucketName}/aggregate/stage_daily/event_date=\${event_date}/`,
        },
        partitionKeys: [{ name: "event_date", type: "string" }],
        storageDescriptor: {
          columns: DAILY_AGGREGATE_COLUMNS,
          compressed: true,
          inputFormat: "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
          outputFormat: "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
          location: `s3://${dataBucket.bucketName}/aggregate/stage_daily/`,
          serdeInfo: {
            serializationLibrary: "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
          },
        },
      },
    });
    dailyStagePerformance.addDependency(database);

    const workGroup = new athena.CfnWorkGroup(this, "AnalyticsWorkGroup", {
      name: "whodoirunlike-processing",
      description: "Cost-bounded private queries over processing telemetry",
      recursiveDeleteOption: false,
      state: "ENABLED",
      workGroupConfiguration: {
        bytesScannedCutoffPerQuery: 1_073_741_824,
        enforceWorkGroupConfiguration: true,
        publishCloudWatchMetricsEnabled: true,
        requesterPaysEnabled: false,
        resultConfiguration: {
          outputLocation: `s3://${dataBucket.bucketName}/athena-results/`,
          encryptionConfiguration: { encryptionOption: "SSE_S3" },
        },
      },
    });

    const dashboardWorkGroup = new athena.CfnWorkGroup(this, "DashboardWorkGroup", {
      name: "whodoirunlike-dashboard",
      description: "Strictly allowlisted, cost-bounded queries for the private analytics UI",
      recursiveDeleteOption: false,
      state: "ENABLED",
      workGroupConfiguration: {
        bytesScannedCutoffPerQuery: 268_435_456,
        enforceWorkGroupConfiguration: true,
        engineVersion: { selectedEngineVersion: "Athena engine version 3" },
        publishCloudWatchMetricsEnabled: true,
        requesterPaysEnabled: false,
        resultConfiguration: {
          outputLocation: `s3://${dataBucket.bucketName}/dashboard-results/`,
          encryptionConfiguration: { encryptionOption: "SSE_S3" },
        },
      },
    });

    const dashboardQueryLogGroup = new logs.LogGroup(this, "DashboardQueryFunctionLogs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const dashboardQueryFunction = new lambda.Function(this, "DashboardQueryFunction", {
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      code: lambda.Code.fromAsset(lambdaSource),
      handler: "dashboard_api.handler",
      timeout: cdk.Duration.seconds(15),
      memorySize: 256,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: dashboardQueryLogGroup,
      environment: {
        ATHENA_DATABASE: databaseName,
        ATHENA_TABLE: tableName,
        ATHENA_WORKGROUP: dashboardWorkGroup.name,
        ATHENA_RESULT_REUSE_MINUTES: "5",
        DASHBOARD_SECRET_ARN: dashboardSecret.secretArn,
        MAX_CLOCK_SKEW_SECONDS: "300",
      },
    });
    dashboardSecret.grantRead(dashboardQueryFunction);
    const dashboardWorkGroupArn = cdk.Stack.of(this).formatArn({
      service: "athena",
      resource: "workgroup",
      resourceName: dashboardWorkGroup.name,
    });
    dashboardQueryFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["athena:GetQueryExecution", "athena:GetQueryResults", "athena:StartQueryExecution"],
        resources: [dashboardWorkGroupArn],
      }),
    );
    dashboardQueryFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["glue:GetDatabase", "glue:GetTable", "glue:GetPartitions"],
        resources: [
          cdk.Stack.of(this).formatArn({ service: "glue", resource: "catalog" }),
          cdk.Stack.of(this).formatArn({
            service: "glue",
            resource: "database",
            resourceName: databaseName,
          }),
          cdk.Stack.of(this).formatArn({
            service: "glue",
            resource: "table",
            resourceName: `${databaseName}/${tableName}`,
          }),
        ],
      }),
    );
    dashboardQueryFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetBucketLocation"],
        resources: [dataBucket.bucketArn],
      }),
    );
    dashboardQueryFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:ListBucket"],
        resources: [dataBucket.bucketArn],
        conditions: {
          StringLike: {
            "s3:prefix": ["validated/*", "dashboard-results/*"],
          },
        },
      }),
    );
    dashboardQueryFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject"],
        resources: [
          dataBucket.arnForObjects("validated/*"),
          dataBucket.arnForObjects("dashboard-results/*"),
        ],
      }),
    );
    dashboardQueryFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:AbortMultipartUpload", "s3:PutObject"],
        resources: [dataBucket.arnForObjects("dashboard-results/*")],
      }),
    );

    const dashboardApiAccessLogs = new logs.LogGroup(this, "DashboardApiAccessLogs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const dashboardApi = new apigateway.RestApi(this, "DashboardApi", {
      description: "HMAC-authenticated API for private processing analytics",
      cloudWatchRole: false,
      deployOptions: {
        stageName: "v1",
        accessLogDestination: new apigateway.LogGroupLogDestination(dashboardApiAccessLogs),
        accessLogFormat: apigateway.AccessLogFormat.jsonWithStandardFields({
          caller: false,
          httpMethod: true,
          ip: true,
          protocol: true,
          requestTime: true,
          resourcePath: true,
          responseLength: true,
          status: true,
          user: false,
        }),
        dataTraceEnabled: false,
        loggingLevel: apigateway.MethodLoggingLevel.ERROR,
        metricsEnabled: true,
        throttlingBurstLimit: 20,
        throttlingRateLimit: 10,
        tracingEnabled: true,
      },
      endpointTypes: [apigateway.EndpointType.REGIONAL],
    });
    // API Gateway has one CloudWatch role per account/region. Reuse the role configured by
    // the ingest API and make fresh-stack creation wait until that account setting exists.
    dashboardApi.deploymentStage.node.addDependency(ingestApi.deploymentStage);
    const dashboardIntegration = new apigateway.LambdaIntegration(dashboardQueryFunction, {
      proxy: true,
    });
    const dashboardQueries = dashboardApi.root.addResource("queries");
    dashboardQueries.addMethod("POST", dashboardIntegration);
    dashboardQueries
      .addResource("{queryExecutionId}")
      .addMethod("GET", dashboardIntegration);

    const aggregateLogGroup = new logs.LogGroup(this, "AggregateFunctionLogs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const aggregateFunction = new lambda.Function(this, "AggregateFunction", {
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      code: lambda.Code.fromAsset(lambdaSource),
      handler: "aggregate.handler",
      timeout: cdk.Duration.seconds(120),
      memorySize: 256,
      tracing: lambda.Tracing.ACTIVE,
      logGroup: aggregateLogGroup,
      environment: {
        EVENT_BUCKET: dataBucket.bucketName,
        ATHENA_DATABASE: databaseName,
        ATHENA_SOURCE_TABLE: tableName,
        ATHENA_WORKGROUP: workGroup.name,
      },
    });
    aggregateFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["athena:GetQueryExecution", "athena:StartQueryExecution", "athena:StopQueryExecution"],
        resources: [
          cdk.Stack.of(this).formatArn({
            service: "athena",
            resource: "workgroup",
            resourceName: workGroup.name,
          }),
        ],
      }),
    );
    aggregateFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["glue:GetDatabase", "glue:GetDatabases", "glue:GetTable", "glue:GetTables", "glue:GetPartitions"],
        resources: [
          cdk.Stack.of(this).formatArn({ service: "glue", resource: "catalog" }),
          cdk.Stack.of(this).formatArn({
            service: "glue",
            resource: "database",
            resourceName: databaseName,
          }),
          cdk.Stack.of(this).formatArn({
            service: "glue",
            resource: "table",
            resourceName: `${databaseName}/${tableName}`,
          }),
        ],
      }),
    );
    aggregateFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetBucketLocation", "s3:ListBucket"],
        resources: [dataBucket.bucketArn],
      }),
    );
    aggregateFunction.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        resources: [
          dataBucket.arnForObjects("validated/*"),
          dataBucket.arnForObjects("aggregate/*"),
          dataBucket.arnForObjects("athena-results/*"),
        ],
      }),
    );
    new events.Rule(this, "DailyAggregateSchedule", {
      description: "Compact yesterday's validated events into indefinitely retained Parquet aggregates",
      schedule: events.Schedule.cron({ minute: "15", hour: "2" }),
      targets: [new eventTargets.LambdaFunction(aggregateFunction, { retryAttempts: 2 })],
    });
    new events.Rule(this, "LateAggregateReconciliationSchedule", {
      description: "Rebuild recent date partitions so delayed outbox events enter long-term history",
      schedule: events.Schedule.cron({ minute: "45", hour: "2" }),
      targets: [
        new eventTargets.LambdaFunction(aggregateFunction, {
          event: events.RuleTargetInput.fromObject({ days_ago: 3, force: true }),
          retryAttempts: 2,
        }),
        new eventTargets.LambdaFunction(aggregateFunction, {
          event: events.RuleTargetInput.fromObject({ days_ago: 7, force: true }),
          retryAttempts: 2,
        }),
      ],
    });

    const deduplicatedCtes = `
ranked AS (
  SELECT *, row_number() OVER (PARTITION BY event_id ORDER BY ingested_at DESC) AS event_rank
  FROM ${databaseName}.${tableName}
  WHERE event_date BETWEEN date_format(current_date - interval '30' day, '%Y-%m-%d')
                       AND date_format(current_date, '%Y-%m-%d')
),
events AS (
  SELECT * FROM ranked WHERE event_rank = 1
)`;

    const stageActiveTimeCtes = `
stage_intervals AS (
  SELECT run_id, attempt_id, sequence, event_id,
         date_add(
           'millisecond',
           -CAST(round(elapsed_seconds * 1000) AS bigint),
           from_iso8601_timestamp(event_time)
         ) AS interval_start_at,
         from_iso8601_timestamp(event_time) AS interval_end_at
  FROM events
  WHERE event_type IN ('stage_completed', 'stage_failed')
    AND elapsed_seconds IS NOT NULL
),
stage_interval_scan AS (
  SELECT *,
         max(interval_end_at) OVER (
           PARTITION BY run_id, attempt_id
           ORDER BY interval_start_at, interval_end_at, sequence, event_id
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
         ) AS prior_max_end_at
  FROM stage_intervals
),
stage_interval_groups AS (
  SELECT *,
         sum(
           CASE WHEN prior_max_end_at IS NULL OR interval_start_at > prior_max_end_at
                THEN 1 ELSE 0 END
         ) OVER (
           PARTITION BY run_id, attempt_id
           ORDER BY interval_start_at, interval_end_at, sequence, event_id
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) AS interval_group
  FROM stage_interval_scan
),
stage_interval_unions AS (
  SELECT run_id, attempt_id, interval_group,
         min(interval_start_at) AS interval_start_at,
         max(interval_end_at) AS interval_end_at
  FROM stage_interval_groups
  GROUP BY run_id, attempt_id, interval_group
),
stage_active_time AS (
  SELECT run_id, attempt_id,
         sum(date_diff('millisecond', interval_start_at, interval_end_at)) / 1000.0 AS observed_stage_seconds
  FROM stage_interval_unions
  GROUP BY run_id, attempt_id
)`;

    const addNamedQuery = (
      id: string,
      props: athena.CfnNamedQueryProps,
    ): athena.CfnNamedQuery => {
      const query = new athena.CfnNamedQuery(this, id, props);
      query.addDependency(workGroup);
      query.addDependency(database);
      return query;
    };

    addNamedQuery("StageLatencyQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "stage_latency_last_30_days",
      description: "Stage p50/p90/p95, throughput, and failure counts",
      queryString: `
WITH ${deduplicatedCtes}
SELECT stage, backend, gpu_type, cold_start, duration_bucket, resolution_bucket,
       count_if(event_type = 'stage_completed') AS completed_samples,
       approx_percentile(CASE WHEN event_type = 'stage_completed' THEN elapsed_seconds END, 0.50) AS p50_seconds,
       approx_percentile(CASE WHEN event_type = 'stage_completed' THEN elapsed_seconds END, 0.90) AS p90_seconds,
       approx_percentile(CASE WHEN event_type = 'stage_completed' THEN elapsed_seconds END, 0.95) AS p95_seconds,
       avg(CASE WHEN event_type = 'stage_completed' AND clip_frame_count > 0 THEN elapsed_seconds * 1000 / clip_frame_count END) AS avg_ms_per_frame,
       avg(CASE WHEN event_type = 'stage_completed' AND clip_duration_seconds > 0 THEN elapsed_seconds / clip_duration_seconds END) AS avg_realtime_factor,
       sum(CASE WHEN event_type = 'stage_failed' THEN 1 ELSE 0 END) AS failures,
       avg(CASE WHEN event_type = 'stage_failed' THEN 1.0 ELSE 0.0 END) AS failure_rate
FROM events
WHERE event_type IN ('stage_completed', 'stage_failed')
GROUP BY stage, backend, gpu_type, cold_start, duration_bucket, resolution_bucket
ORDER BY p95_seconds DESC`,
    });

    addNamedQuery("SpanLatencyQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "span_latency_last_30_days",
      description: "Substage p50/p90/p95 to explain expensive stages",
      queryString: `
WITH ${deduplicatedCtes},
span_attempt_totals AS (
  SELECT run_id, attempt_id, stage, span, backend, gpu_type, cold_start,
         duration_bucket, resolution_bucket, artifact_type,
         max(clip_frame_count) AS clip_frame_count,
         sum(elapsed_seconds) AS total_span_seconds,
         count(*) AS occurrences,
         sum(CASE WHEN event_type = 'span_failed' THEN 1 ELSE 0 END) AS failed_occurrences
  FROM events
  WHERE event_type IN ('span_completed', 'span_failed')
  GROUP BY run_id, attempt_id, stage, span, backend, gpu_type, cold_start,
           duration_bucket, resolution_bucket, artifact_type
)
SELECT stage, span, backend, gpu_type, cold_start, duration_bucket, resolution_bucket,
       artifact_type,
       count_if(failed_occurrences = 0) AS completed_attempts,
       sum(occurrences) AS span_occurrences,
       approx_percentile(CASE WHEN failed_occurrences = 0 THEN total_span_seconds END, 0.50) AS p50_seconds_per_attempt,
       approx_percentile(CASE WHEN failed_occurrences = 0 THEN total_span_seconds END, 0.90) AS p90_seconds_per_attempt,
       approx_percentile(CASE WHEN failed_occurrences = 0 THEN total_span_seconds END, 0.95) AS p95_seconds_per_attempt,
       avg(CASE WHEN failed_occurrences = 0 AND clip_frame_count > 0 THEN total_span_seconds * 1000 / clip_frame_count END) AS avg_ms_per_frame,
       sum(failed_occurrences) AS failed_occurrences
FROM span_attempt_totals
GROUP BY stage, span, backend, gpu_type, cold_start, duration_bucket, resolution_bucket,
         artifact_type
ORDER BY p95_seconds_per_attempt DESC`,
    });

    addNamedQuery("AttemptLatencyQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "attempt_result_ready_latency_last_30_days",
      description: "User-facing Result Ready latency distribution",
      queryString: `
WITH ${deduplicatedCtes}
SELECT environment, gpu_type, cold_start, duration_bucket, resolution_bucket,
       count(*) AS attempts,
       approx_percentile(elapsed_seconds, 0.50) AS p50_seconds,
       approx_percentile(elapsed_seconds, 0.90) AS p90_seconds,
       approx_percentile(elapsed_seconds, 0.95) AS p95_seconds
FROM events
WHERE event_type = 'result_ready'
GROUP BY environment, gpu_type, cold_start, duration_bucket, resolution_bucket
ORDER BY p95_seconds DESC`,
    });

    addNamedQuery("FailureQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "processing_failures_last_30_days",
      description: "Failures by stage/span/error class without raw exception text",
      queryString: `
WITH ${deduplicatedCtes}
SELECT stage, span, error_class, error_code, count(*) AS failures,
       approx_percentile(elapsed_seconds, 0.50) AS p50_time_to_failure_seconds,
       approx_percentile(elapsed_seconds, 0.95) AS p95_time_to_failure_seconds,
       max(event_time) AS most_recent
FROM events
WHERE event_type IN ('span_failed', 'stage_failed', 'attempt_failed')
GROUP BY stage, span, error_class, error_code
ORDER BY failures DESC, most_recent DESC`,
    });

    addNamedQuery("QueueLatencyQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "enqueue_queue_and_model_load_latency_last_30_days",
      description: "Separates orchestration, RunPod queue, and cold model-load latency",
      queryString: `
WITH ${deduplicatedCtes}
SELECT stage, span, backend, gpu_type, cold_start, timing_basis,
       count(*) AS samples,
       approx_percentile(elapsed_seconds, 0.50) AS p50_seconds,
       approx_percentile(elapsed_seconds, 0.90) AS p90_seconds,
       approx_percentile(elapsed_seconds, 0.95) AS p95_seconds,
       avg(elapsed_seconds) AS average_seconds
FROM events
WHERE (event_type = 'stage_completed' AND stage IN ('source_ingest', 'processor_enqueue'))
   OR (event_type = 'stage_completed' AND stage = 'processor_queue'
       AND timing_basis IN ('runpod_delay_time', 'worker_dispatch_to_start_estimate'))
   OR (event_type = 'span_completed' AND span = 'model_load')
GROUP BY stage, span, backend, gpu_type, cold_start, timing_basis
ORDER BY p95_seconds DESC`,
    });

    addNamedQuery("AttemptBreakdownQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "attempt_stage_breakdown_last_30_days",
      description: "One row per attempt with overlap-aware active stage time",
      queryString: `
WITH ${deduplicatedCtes},
${stageActiveTimeCtes},
attempts AS (
  SELECT run_id, attempt_id, max(attempt_number) AS attempt_number,
       max(duration_bucket) AS duration_bucket,
       max(resolution_bucket) AS resolution_bucket,
       max(gpu_type) AS gpu_type,
       max(CASE WHEN stage = 'source_ingest' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS source_ingest_seconds,
       max(CASE WHEN stage = 'processor_enqueue' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS processor_enqueue_seconds,
       max(CASE WHEN stage = 'processor_queue' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS processor_queue_seconds,
       max(CASE WHEN stage = 'source_download' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS source_download_seconds,
       max(CASE WHEN stage = 'run_preparation' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS run_preparation_seconds,
       max(CASE WHEN stage = 'target_tracking' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS target_tracking_seconds,
       max(CASE WHEN stage = 'runner_mask' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS runner_mask_seconds,
       max(CASE WHEN stage = 'pose_sequence' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS pose_sequence_seconds,
       max(CASE WHEN stage = 'densepose_body_map' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS densepose_seconds,
       max(CASE WHEN stage = 'fused_form_signal' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS fusion_seconds,
       max(CASE WHEN stage = 'form_feature_compilation' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS feature_compilation_seconds,
       max(CASE WHEN stage = 'artifact_table_export' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS table_export_seconds,
       max(CASE WHEN stage = 'quality_control' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS quality_control_seconds,
       max(CASE WHEN stage = 'analysis_complete' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS analysis_finalize_seconds,
       max(CASE WHEN stage = 'artifact_publish' AND event_type = 'stage_completed' THEN elapsed_seconds END) AS artifact_publish_seconds,
       max(CASE WHEN event_type = 'result_ready' THEN elapsed_seconds END) AS result_ready_seconds,
       max(CASE WHEN event_type = 'analysis_completed' THEN elapsed_seconds END) AS analysis_complete_seconds,
       max(CASE WHEN event_type = 'attempt_completed' THEN elapsed_seconds END) AS attempt_complete_seconds,
       max(CASE WHEN event_type = 'attempt_failed' THEN elapsed_seconds END) AS attempt_failed_seconds,
       max_by(stage, elapsed_seconds) FILTER (WHERE event_type = 'stage_completed') AS top_bottleneck_stage,
       max(elapsed_seconds) FILTER (WHERE event_type = 'stage_completed') AS top_bottleneck_seconds
  FROM events
  GROUP BY run_id, attempt_id
)
SELECT a.*,
       coalesce(s.observed_stage_seconds, 0.0) AS observed_stage_seconds,
       greatest(
         greatest(
           coalesce(a.attempt_complete_seconds, 0),
           coalesce(a.result_ready_seconds, 0),
           coalesce(a.attempt_failed_seconds, 0)
         ) - coalesce(s.observed_stage_seconds, 0.0),
         0
       ) AS unattributed_to_attempt_complete_seconds
FROM attempts a
LEFT JOIN stage_active_time s
  ON a.run_id = s.run_id AND a.attempt_id = s.attempt_id
ORDER BY result_ready_seconds DESC NULLS LAST`,
    });

    addNamedQuery("MilestoneGapQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "result_ready_vs_analysis_complete_last_30_days",
      description: "Measures publish lag from analysis completion to the user-visible result",
      queryString: `
WITH ${deduplicatedCtes},
attempts AS (
  SELECT run_id, attempt_id,
         max(CASE WHEN event_type = 'result_ready' THEN elapsed_seconds END) AS result_ready_seconds,
         max(CASE WHEN event_type = 'analysis_completed' THEN elapsed_seconds END) AS analysis_complete_seconds
  FROM events
  GROUP BY run_id, attempt_id
)
SELECT run_id, attempt_id, result_ready_seconds, analysis_complete_seconds,
       result_ready_seconds - analysis_complete_seconds AS publish_after_analysis_seconds
FROM attempts
WHERE result_ready_seconds IS NOT NULL OR analysis_complete_seconds IS NOT NULL
ORDER BY greatest(coalesce(result_ready_seconds, 0), coalesce(analysis_complete_seconds, 0)) DESC`,
    });

    addNamedQuery("StalledAttemptsQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "stalled_attempts",
      description: "Attempts with no terminal event and no telemetry for at least ten minutes",
      queryString: `
WITH ${deduplicatedCtes},
attempts AS (
  SELECT run_id, attempt_id,
         min(event_time) AS first_event_time,
         max(event_time) AS last_event_time,
         max(CASE WHEN event_type IN ('attempt_completed', 'attempt_failed') THEN 1 ELSE 0 END) AS terminal_events,
         max(CASE WHEN stage IN ('processor_enqueue', 'processor_queue') THEN 1 ELSE 0 END) AS processing_was_requested,
         max_by(stage, event_time) AS last_stage,
         max_by(span, event_time) AS last_span,
         max_by(event_type, event_time) AS last_event_type
  FROM events
  GROUP BY run_id, attempt_id
)
SELECT *
FROM attempts
WHERE terminal_events = 0
  AND processing_was_requested = 1
  AND from_iso8601_timestamp(last_event_time) < current_timestamp - interval '10' minute
ORDER BY last_event_time ASC`,
    });

    addNamedQuery("DailyAggregateQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "daily_stage_performance",
      description: "Daily aggregate source query for the indefinite aggregate zone",
      queryString: `
WITH ${deduplicatedCtes}
SELECT event_date, event_type, stage, span, backend, gpu_type, cold_start,
       duration_bucket, resolution_bucket,
       count(*) AS samples,
       approx_percentile(elapsed_seconds, 0.50) AS p50_seconds,
       approx_percentile(elapsed_seconds, 0.90) AS p90_seconds,
       approx_percentile(elapsed_seconds, 0.95) AS p95_seconds,
       avg(elapsed_seconds) AS average_seconds,
       sum(CASE WHEN event_type IN ('span_failed', 'stage_failed', 'attempt_failed') THEN 1 ELSE 0 END) AS failures
FROM events
WHERE event_type IN (
  'stage_completed', 'stage_failed', 'span_completed', 'span_failed',
  'result_ready', 'analysis_completed', 'attempt_completed', 'attempt_failed'
)
GROUP BY event_date, event_type, stage, span, backend, gpu_type, cold_start,
         duration_bucket, resolution_bucket
ORDER BY event_date DESC, p95_seconds DESC`,
    });

    addNamedQuery("DailyAggregateHistoryQuery", {
      database: databaseName,
      workGroup: workGroup.name,
      name: "daily_stage_performance_history",
      description: "Indefinite daily Parquet history for performance analysis",
      queryString: `
SELECT event_date, event_type, stage, span, backend, gpu_type, cold_start,
       duration_bucket, resolution_bucket,
       samples, p50_seconds, p90_seconds, p95_seconds, average_seconds, failures
FROM ${databaseName}.${aggregateTableName}
WHERE event_date BETWEEN date_format(current_date - interval '365' day, '%Y-%m-%d')
                     AND date_format(current_date, '%Y-%m-%d')
ORDER BY event_date DESC, p95_seconds DESC`,
    });

    const operationsDashboard = new cloudwatch.Dashboard(this, "OperationsDashboard", {
      dashboardName: "whodoirunlike-analytics-pipeline",
    });
    operationsDashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: "Telemetry ingestion and validation",
        left: [
          ingestFunction.metricInvocations(),
          new cloudwatch.Metric({
            namespace: analyticsMetricNamespace,
            metricName: "IngestRejected",
            statistic: "Sum",
          }),
        ],
        right: [
          new cloudwatch.Metric({
            namespace: analyticsMetricNamespace,
            metricName: "IngestUnavailable",
            statistic: "Sum",
          }),
          new cloudwatch.Metric({
            namespace: analyticsMetricNamespace,
            metricName: "EventLakeWriteFailure",
            statistic: "Sum",
          }),
        ],
      }),
      new cloudwatch.GraphWidget({
        title: "Queue depth, age, and dead letters",
        left: [
          telemetryQueue.metricApproximateNumberOfMessagesVisible(),
          telemetryQueue.metricApproximateAgeOfOldestMessage(),
        ],
        right: [deadLetterQueue.metricApproximateNumberOfMessagesVisible()],
      }),
    );
    new cloudwatch.Alarm(this, "DeadLetterAlarm", {
      metric: deadLetterQueue.metricApproximateNumberOfMessagesVisible({ period: cdk.Duration.minutes(5) }),
      evaluationPeriods: 1,
      threshold: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      alarmDescription: "At least one processing telemetry event could not be validated or stored",
    });

    new cdk.CfnOutput(this, "TelemetryIngestUrl", {
      value: `${ingestApi.url}events`,
      description: "Set as AWS_ANALYTICS_INGEST_URL on the Cloudflare Worker",
    });
    new cdk.CfnOutput(this, "TelemetrySecretArn", {
      value: ingestSecret.secretArn,
      description: "Retrieve once and store as the Worker AWS_ANALYTICS_SHARED_SECRET",
    });
    new cdk.CfnOutput(this, "EventLakeBucket", { value: dataBucket.bucketName });
    new cdk.CfnOutput(this, "AthenaDatabase", { value: databaseName });
    new cdk.CfnOutput(this, "AthenaWorkGroup", { value: workGroup.name });
    new cdk.CfnOutput(this, "DashboardApiUrl", {
      value: `${dashboardApi.url}queries`,
      description: "Store only on the private dashboard Worker; never expose this endpoint to browsers",
    });
    new cdk.CfnOutput(this, "DashboardSecretArn", {
      value: dashboardSecret.secretArn,
      description: "Retrieve once and store as the dashboard Worker's AWS_DASHBOARD_SHARED_SECRET",
    });
    new cdk.CfnOutput(this, "DashboardAthenaWorkGroup", {
      value: dashboardWorkGroup.name,
    });
    new cdk.CfnOutput(this, "PrivateOperationsDashboard", {
      value: operationsDashboard.dashboardName,
    });
  }
}
