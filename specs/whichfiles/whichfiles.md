# Change Analysis Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

You will review the provided system to identify all files that would need modifications to implement the following change: 
    <changes> description: '<description>' </changes>

## Mid-Level Objective

- Analyze codebase to identify affected files
- Document required modifications and their justifications
- Assess architectural impacts and dependencies
- Generate clarifying questions for ambiguous requirements

<thought_process>
1. First identify direct references to the topic in the description
2. Then trace dependencies through system interfaces
3. Map out affected codebase and documentation sections
</thought_process>

## Implementation Notes
- Follow YAML specification for output file
- Include clear justification for each identified file

## Context
'<context>'

## Low-Level Tasks
> Ordered from start to finish

1. Generate Change Analysis Document
in a ```yaml section, CREATE tochange.yaml contents:
    Format the file with the following sections:
    - Files Requiring Updates (key Files_Requiring_Updates)
      - For each file:
        - path
        - Reason for modification
        - Spec (not implementation details) of changes needed (key spec_of_changes)
        - Dependencies affected
    - Architectural Impact Assessment
    - Questions for Clarification
    
    The content should be based on analyzing the provided context against the change description.
