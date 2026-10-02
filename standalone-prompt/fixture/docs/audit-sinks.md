# Audit delivery

The service records operational events at a package boundary so deployments can choose an appropriate destination. Event labels arrive from command-line flags, hooks, and external integrations, so they are not always formatted consistently.

Metrics collection and archival export use separate protocols. Changes to the audit boundary should avoid coupling those components to event persistence details.
