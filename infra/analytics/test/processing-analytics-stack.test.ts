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
  output.resourceCountIs("AWS::ApiGateway::RestApi", 1);
  output.resourceCountIs("AWS::SQS::Queue", 2);
  output.resourceCountIs("AWS::Lambda::Function", 3);
  output.resourceCountIs("AWS::S3::Bucket", 1);
  output.resourceCountIs("AWS::Glue::Database", 1);
  output.resourceCountIs("AWS::Glue::Table", 2);
  output.resourceCountIs("AWS::Athena::WorkGroup", 1);
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
