#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";

import { ProcessingAnalyticsStack } from "../lib/processing-analytics-stack.js";

const app = new cdk.App();
const stackName = process.env.WDIRL_ANALYTICS_STACK_NAME || "WhoDoIRunLikeAnalytics";

new ProcessingAnalyticsStack(app, stackName, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || process.env.AWS_REGION || "us-east-1",
  },
  description: "Private processing-observability event lake for Who Do I Run Like",
});
