# Subagent-Driven Development (`superpowers:subagent-driven-development`)

## Overview
Coordinates multiple parallel tasks using background agents to implement a structured plan. This skill solves context drift by isolating each task into a fresh agent session.

## The Workflow
1. **Decompose**: Identify independent tasks in the plan.
2. **Dispatch**: Use `invoke_agent` for each task with specific context and scope.
3. **Review**: Perform spec compliance and code quality reviews.
4. **Integrate**: Resolve conflicts and run full test suites.

## Mandate
Always use specialized subagents for repetitive or high-volume tasks.
