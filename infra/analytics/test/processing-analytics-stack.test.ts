import assert from "node:assert/strict";
import test from "node:test";

import * as cdk from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";

import { ProcessingAnalyticsStack } from "../lib/processing-analytics-stack.js";

function template(): Template {
  const app = new cdk.App();
  const stack = new ProcessingAnalyticsStack(app, "TestAnalytics", {
    env: { account: "111122223333", region: "us-east-1" },
  });
  return Template.fromStack(stack);
}

test("builds the complete private event-lake path", () => {
  const output = template();
  output.resourceCountIs("AWS::ApiGateway::RestApi", 2);
  output.resourceCountIs("AWS::ApiGateway::Authorizer", 0);
  output.resourceCountIs("AWS::ApiGateway::Account", 1);
  output.resourceCountIs("AWS::SQS::Queue", 2);
  output.resourceCountIs("AWS::Lambda::Function", 4);
  output.resourceCountIs("AWS::SecretsManager::Secret", 2);
  output.resourceCountIs("AWS::S3::Bucket", 1);
  output.resourceCountIs("AWS::Glue::Database", 1);
  output.resourceCountIs("AWS::Glue::Table", 2);
  output.resourceCountIs("AWS::Athena::WorkGroup", 2);
  output.resourceCountIs("AWS::Athena::NamedQuery", 10);
  output.resourceCountIs("AWS::CloudWatch::Dashboard", 1);
  output.resourceCountIs("AWS::Logs::MetricFilter", 3);
  output.resourceCountIs("AWS::Events::Rule", 2);
});

test("keeps processing metadata encrypted, private, and lifecycle bounded", () => {
  const output = template();
  output.hasResourceProperties("AWS::S3::Bucket", {
    PublicAccessBlockConfiguration: {
      BlockPublicAcls: true,
      BlockPublicPolicy: true,
      IgnorePublicAcls: true,
      RestrictPublicBuckets: true,
    },
    LifecycleConfiguration: {
      Rules: Match.arrayWith([
        Match.objectLike({ Id: "expire-raw-events", ExpirationInDays: 90 }),
        Match.objectLike({ Id: "expire-validated-events", ExpirationInDays: 365 }),
        Match.objectLike({
          Id: "expire-validated-progress",
          ExpirationInDays: 90,
          TagFilters: [{ Key: "event-class", Value: "progress" }],
        }),
        Match.objectLike({ Id: "expire-athena-results", ExpirationInDays: 30 }),
        Match.objectLike({ Id: "expire-dashboard-results", ExpirationInDays: 7 }),
      ]),
    },
  });
  output.hasResourceProperties("AWS::SecretsManager::Secret", {
    GenerateSecretString: { ExcludePunctuation: true, PasswordLength: 64 },
  });
  for (const resourceType of [
    "AWS::S3::Bucket",
    "AWS::SQS::Queue",
    "AWS::SecretsManager::Secret",
  ]) {
    for (const resource of Object.values(output.findResources(resourceType))) {
      assert.equal(resource.DeletionPolicy, "RetainExceptOnCreate");
      assert.equal(resource.UpdateReplacePolicy, "Retain");
    }
  }
});

test("uses ordered deduplicated delivery and partial batch failures", () => {
  const output = template();
  const queues = output.findResources("AWS::SQS::Queue");
  const sourceQueue = Object.values(queues).find(
    (resource) => resource.Properties?.VisibilityTimeout === 210,
  );
  assert.ok(sourceQueue, "source FIFO queue should have a 210-second visibility timeout");
  assert.equal(sourceQueue.Properties?.FifoQueue, true);
  assert.equal(sourceQueue.Properties?.RedrivePolicy?.maxReceiveCount, 5);

  output.hasResourceProperties("AWS::Lambda::EventSourceMapping", {
    BatchSize: 10,
    FunctionResponseTypes: ["ReportBatchItemFailures"],
  });
});

test("cost bounds Athena and schedules indefinite daily aggregates", () => {
  const output = template();
  output.hasResourceProperties("AWS::Athena::WorkGroup", {
    WorkGroupConfiguration: Match.objectLike({
      BytesScannedCutoffPerQuery: 1_073_741_824,
      EnforceWorkGroupConfiguration: true,
      PublishCloudWatchMetricsEnabled: true,
    }),
  });
  output.hasResourceProperties("AWS::Events::Rule", {
    ScheduleExpression: "cron(15 2 * * ? *)",
    State: "ENABLED",
  });
  output.hasResourceProperties("AWS::Events::Rule", {
    ScheduleExpression: "cron(45 2 * * ? *)",
    State: "ENABLED",
    Targets: Match.arrayWith([
      Match.objectLike({ Input: '{"days_ago":3,"force":true}' }),
      Match.objectLike({ Input: '{"days_ago":7,"force":true}' }),
    ]),
  });
});

test("queries expose complete attempt attribution and per-attempt span totals", () => {
  const output = template();
  const queries = Object.values(output.findResources("AWS::Athena::NamedQuery"));
  for (const query of queries) {
    const dependencies = Array.isArray(query.DependsOn)
      ? query.DependsOn
      : [query.DependsOn].filter(Boolean);
    assert.ok(
      dependencies.includes("AnalyticsWorkGroup"),
      "every named query must wait for its Athena workgroup",
    );
    assert.ok(
      dependencies.includes("AnalyticsDatabase"),
      "every named query must wait for its Glue database",
    );
  }
  const queryByName = (name: string): string => {
    const resource = queries.find((candidate) => candidate.Properties?.Name === name);
    assert.ok(resource, `missing Athena named query: ${name}`);
    return String(resource.Properties?.QueryString);
  };

  const attemptQuery = queryByName("attempt_stage_breakdown_last_30_days");
  for (const stage of [
    "source_download",
    "run_preparation",
    "form_feature_compilation",
    "artifact_table_export",
    "quality_control",
    "artifact_publish",
  ]) {
    assert.match(attemptQuery, new RegExp(stage));
  }
  assert.match(attemptQuery, /unattributed_to_attempt_complete_seconds/);
  assert.match(attemptQuery, /top_bottleneck_stage/);

  const spanQuery = queryByName("span_latency_last_30_days");
  assert.match(spanQuery, /sum\(elapsed_seconds\) AS total_span_seconds/);
  assert.match(spanQuery, /artifact_type/);

  const stalledQuery = queryByName("stalled_attempts");
  assert.match(stalledQuery, /processing_was_requested = 1/);
});

test("dashboard queries run in a separate low-scan Athena workgroup", () => {
  const output = template();
  output.hasResourceProperties("AWS::Athena::WorkGroup", {
    Name: "whodoirunlike-dashboard",
    WorkGroupConfiguration: Match.objectLike({
      BytesScannedCutoffPerQuery: 268_435_456,
      EnforceWorkGroupConfiguration: true,
      EngineVersion: { SelectedEngineVersion: "Athena engine version 3" },
      ResultConfiguration: Match.objectLike({
        EncryptionConfiguration: { EncryptionOption: "SSE_S3" },
        OutputLocation: Match.anyValue(),
      }),
    }),
  });

  const workgroups = Object.values(output.findResources("AWS::Athena::WorkGroup"));
  const dashboard = workgroups.find(
    (resource) => resource.Properties?.Name === "whodoirunlike-dashboard",
  );
  assert.ok(dashboard);
  assert.match(JSON.stringify(dashboard), /dashboard-results\//);
});

test("routes both dashboard API methods through the HMAC-verifying query Lambda", () => {
  const output = template();
  const methods = Object.values(output.findResources("AWS::ApiGateway::Method")).filter(
    (resource) =>
      ["POST", "GET"].includes(resource.Properties?.HttpMethod) &&
      JSON.stringify(resource.Properties?.Integration).includes("DashboardQueryFunction"),
  );
  assert.equal(methods.length, 2);
  assert.deepEqual(
    new Set(methods.map((resource) => resource.Properties?.HttpMethod)),
    new Set(["POST", "GET"]),
  );
  for (const method of methods) {
    assert.equal(method.Properties?.AuthorizationType, "NONE");
    assert.equal(method.Properties?.AuthorizerId, undefined);
    assert.equal(method.Properties?.Integration?.Type, "AWS_PROXY");
  }

  output.hasResourceProperties("AWS::Lambda::Function", {
    Handler: "dashboard_api.handler",
    Environment: {
      Variables: Match.objectLike({
        DASHBOARD_SECRET_ARN: { Ref: Match.stringLikeRegexp("DashboardHmacSecret") },
        MAX_CLOCK_SKEW_SECONDS: "300",
      }),
    },
  });
  output.hasResourceProperties("AWS::SecretsManager::Secret", {
    Description: "HMAC secret used only by the private dashboard Cloudflare Worker",
    GenerateSecretString: { ExcludePunctuation: true, PasswordLength: 64 },
  });

  const stages = output.findResources("AWS::ApiGateway::Stage");
  const dashboardStage = Object.entries(stages).find(([logicalId]) =>
    logicalId.startsWith("DashboardApiDeploymentStage"),
  );
  assert.ok(dashboardStage, "dashboard API stage is missing");
  assert.ok(
    (dashboardStage[1].DependsOn || []).some((dependency: string) =>
      dependency.startsWith("IngestApiDeploymentStage"),
    ),
    "dashboard stage must wait for the shared API Gateway CloudWatch account role",
  );
});

test("gives the dashboard query Lambda only fixed table and result-prefix access", () => {
  const output = template();
  output.hasResourceProperties("AWS::Lambda::Function", {
    Handler: "dashboard_api.handler",
    Environment: {
      Variables: Match.objectLike({
        ATHENA_DATABASE: "whodoirunlike_analytics",
        ATHENA_TABLE: "processing_events",
        ATHENA_WORKGROUP: "whodoirunlike-dashboard",
        ATHENA_RESULT_REUSE_MINUTES: "1",
      }),
    },
  });

  const policies = Object.values(output.findResources("AWS::IAM::Policy"));
  const dashboardPolicy = policies.find((resource) =>
    String(resource.Properties?.PolicyName).startsWith("DashboardQueryFunction"),
  );
  assert.ok(dashboardPolicy, "dashboard query role policy is missing");
  const document = JSON.stringify(dashboardPolicy.Properties?.PolicyDocument);
  assert.match(document, /athena:GetQueryResults/);
  assert.match(document, /workgroup\/whodoirunlike-dashboard/);
  assert.match(document, /validated\/\*/);
  assert.match(document, /dashboard-results\/\*/);
  assert.doesNotMatch(document, /raw\/\*/);
  assert.doesNotMatch(document, /aggregate\/\*/);
  assert.doesNotMatch(document, /s3:DeleteObject/);
});

test("does not provision QuickSight", () => {
  const output = template();
  assert.equal(Object.keys(output.findResources("AWS::QuickSight::Dashboard")).length, 0);
  assert.equal(Object.keys(output.findResources("AWS::QuickSight::DataSource")).length, 0);
});

test("exports only the server-side dashboard connection values", () => {
  const output = template();
  output.hasOutput("DashboardApiUrl", {});
  output.hasOutput("DashboardSecretArn", {});
  output.hasOutput("DashboardAthenaWorkGroup", { Value: "whodoirunlike-dashboard" });
  assert.equal(Object.keys(output.findOutputs("DashboardAccessHeader")).length, 0);
});
