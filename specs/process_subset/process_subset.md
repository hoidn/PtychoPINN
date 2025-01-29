<role>Software architect</role>
## Context

## High-Level Objective

Write a spec prompt according to the given description and Q&A clarifications. 

<instructions>

- in <thinking> tags, analyze the following:
   - Review any open questions in the <high-level objective>. If there are any, address them 
   - How this affects the requested changes
   - Key points from the Q&A discussion
   - approach for the <high-level objective>
   - Which data types and structures need to be added, removed, modified
   - Use <scratchpad> tags to quote relevant code sections and <brainstorm> tags to consider alternative approaches before commiting to a design choice
- For each file we will modify, note dependencies and potential impacts
- Then draft the necessary structural changes and behavioral changes in the format of a spec prompt, as documented in the <spec prompt guide>

</instructions>

### Beginning Context

- tochange.yaml, containing the files that need to be modified and other relevant information
- Description of TODO / <high-level objective>: see <desc> 
- Questions and answers: see <quest>
- Spec prompt guide: see <guide>

### Ending Context

- taskspec.md, containing a well-formed spec prompt documenting the changes necessary to implement <high-level objective>

<output_format>
- Include type hints when possible (type-driven development)
  - BAD: 'CREATE a new function create_gist'
  - GOOD: 'CREATE def create_gist(gist: Gist) -> dict or throw'
  - the spec prompt should instruct the changes necessary to implement the <high-level objective>. It should specify which components need to be added, removed, modified, and what the behavioral changes are. 
  - The spec prompt should be enclosed in <taskspec> tags.

</output_format>

<desc>
'<description>'
</desc>

<quest>
'<questions>'
</quest>

<guide>
'<spec_prompt_guide>'
</guide>

