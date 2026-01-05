# AGENT INSTRUCTIONS - IMPORTANT REMINDER

## CRITICAL RULE: NO CODE CHANGES WITHOUT APPROVED PLAN

**NEVER make changes (code, tests, configuration, build scripts, documentation, deletions, formatting) without first presenting a plan and getting explicit user approval.**

### Process to Follow:
1. **Analyze the issue** - Understand what needs to be fixed
2. **Create a detailed plan** - Explain what changes will be made and why
3. **Present the plan to user** - Wait for explicit approval ("approved", "yes", "do it", etc.)
4. **ONLY AFTER approval** - Implement the code changes
5. **Never skip steps** - Always follow this process

## CLARIFICATION RULE: STOP WHEN UNCERTAIN

If any requirement is ambiguous or there are multiple reasonable implementation choices:
1. Present the options and tradeoffs
2. Ask the user to choose
3. Do not implement until the user confirms

## COMMENT AND DOCUMENTATION POLICY

Comment and documentation updates are allowed when they improve correctness and maintainability.

Requirements:
- Production tone only
- No emojis
- No celebratory or vague wording in comments (examples: "Fixed", "Enhanced", "Quick hack")
- Comments must explain intent or constraints, not narrate edits

## CODING PRINCIPLES TO FOLLOW:

### 1. Use Simple Solutions
- **Prefer simple over complex** - Choose the simplest solution that works
- **Avoid over-engineering** - Don't add unnecessary complexity
- **Minimal changes** - Make the smallest change possible to fix the issue
- **Clear and readable** - Write code that is easy to understand
- **Avoid premature optimization** - Don't optimize unless it's actually needed

### 2. Follow SOLID Principles
- **S**ingle Responsibility - Each class/function does one thing well
- **O**pen/Closed - Open for extension, closed for modification
- **L**iskov Substitution - Derived classes can replace base classes
- **I**nterface Segregation - Small, focused interfaces
- **D**ependency Inversion - Depend on abstractions, not concretions

### 3. Maintain Separation of Concerns
- **Clear module boundaries** - Each module/package has a distinct purpose
- **Don't mix responsibilities** - Audio processing stays separate from business logic
- **Interface boundaries are sacred** - Don't bypass abstractions or create tight coupling
- **Data flow is unidirectional** - Audio → ASR → Processing → Response (no circular dependencies)
- **Configuration vs implementation** - Keep config separate from business logic
- **UI concerns stay in UI** - Don't mix server logic with client-side display logic

**Examples in this codebase:**
- `src/asr/` - Handles only speech recognition concerns
- `src/gateway/` - Manages only utterance processing and WebRTC concerns
- `src/constants/` - Contains only shared constants, no business logic
- `tests/` - Separate test concerns from implementation

**Violations to avoid:**
- Don't put audio processing logic in utterance management
- Don't mix UI state management with ASR provider logic
- Don't create circular dependencies between modules
- Don't bypass the factory pattern for ASR providers

### Previous Violations:
- Made unauthorized changes to google_speech_v2.py model mapping
- Updated Makefile without approval
- Modified recognition config without permission
- Over-engineered solutions instead of simple fixes

### Consequences:
- User explicitly stated: "Do not make further code changes without my approving a plan ever again"
- This instruction must be followed 100% of the time

### Current State:
- Google Speech-to-Text v2 provider is using latest_long model
- Makefile shows model mapping (gemini-2.5-flash -> latest_long)
- Enhanced recognition config with phrase hints added
- Server running with improved configuration

### Next Steps (ONLY WITH APPROVAL):
- Test the current configuration
- If transcription quality is still poor, present SIMPLE options for improvements
- Any additional changes require a detailed plan first
- Always choose the SIMPLEST solution that works

**REMEMBER: ALWAYS GET APPROVAL BEFORE MAKING CODE CHANGES!**
**REMEMBER: USE SIMPLE SOLUTIONS AND FOLLOW SOLID PRINCIPLES!**
