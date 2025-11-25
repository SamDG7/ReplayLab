import re
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# -----------------------------------------------------------
# 1. ACTION LIBRARY
# -----------------------------------------------------------

ACTIONS = {
    "analysis": [
        {"action": "Identify core entities in the prompt", "quality": "good"},
        {"action": "Break down the problem into subcomponents", "quality": "good"},
        {"action": "Perform multi-step reasoning on the problem", "quality": "good"},
        {"action": "Map relationships between different concepts", "quality": "good"},
        {"action": "Make broad assumptions without justification", "quality": "error"},
        {"action": "Overfocus on irrelevant keywords", "quality": "suboptimal"},
    ],
    "writing": [
        {"action": "Draft a structured outline", "quality": "good"},
        {"action": "Write a high-level summary", "quality": "good"},
        {"action": "Finalize a polished explanation", "quality": "good"},
        {"action": "Adapt tone and style to target audience", "quality": "good"},
        {"action": "Jump to answer without any analysis", "quality": "error"},
        {"action": "Repeat previously written content", "quality": "suboptimal"},
    ],
    "research": [
        {"action": "Generate a search query from prompt intent", "quality": "good"},
        {"action": "Extract relevant external information", "quality": "good"},
        {"action": "Verify consistency of gathered evidence", "quality": "good"},
        {"action": "Cross-reference multiple sources", "quality": "good"},
        {"action": "Hallucinate unsupported facts", "quality": "error"},
        {"action": "Use overly generic external info", "quality": "suboptimal"},
    ],
    "math": [
        {"action": "Translate problem into equations", "quality": "good"},
        {"action": "Check each computational step for correctness", "quality": "good"},
        {"action": "Compute final result with justification", "quality": "good"},
        {"action": "Verify answer against constraints", "quality": "good"},
        {"action": "Perform incorrect symbolic manipulation", "quality": "error"},
        {"action": "Skip validation of derived formulas", "quality": "suboptimal"},
    ],
    "planning": [
        {"action": "Create an execution plan for solving the prompt", "quality": "good"},
        {"action": "Evaluate alternative solution paths", "quality": "good"},
        {"action": "Refine approach based on new insights", "quality": "good"},
        {"action": "Identify dependencies and prerequisites", "quality": "good"},
        {"action": "Loop endlessly in planning state", "quality": "error"},
        {"action": "Choose overly complex plan for simple task", "quality": "suboptimal"},
    ],
    "coding": [
        {"action": "Analyze requirements and constraints", "quality": "good"},
        {"action": "Design data structures and algorithms", "quality": "good"},
        {"action": "Implement core functionality", "quality": "good"},
        {"action": "Write unit tests for edge cases", "quality": "good"},
        {"action": "Introduce syntax errors or type mismatches", "quality": "error"},
        {"action": "Skip error handling for critical paths", "quality": "suboptimal"},
    ],
    "debugging": [
        {"action": "Reproduce the issue consistently", "quality": "good"},
        {"action": "Isolate the problematic component", "quality": "good"},
        {"action": "Trace execution flow and state changes", "quality": "good"},
        {"action": "Apply targeted fix and verify resolution", "quality": "good"},
        {"action": "Make random changes without understanding", "quality": "error"},
        {"action": "Fix symptoms instead of root cause", "quality": "suboptimal"},
    ],
    "refactoring": [
        {"action": "Identify code smells and technical debt", "quality": "good"},
        {"action": "Extract reusable components", "quality": "good"},
        {"action": "Improve naming and structure", "quality": "good"},
        {"action": "Maintain existing functionality", "quality": "good"},
        {"action": "Break working code during refactor", "quality": "error"},
        {"action": "Over-engineer simple solutions", "quality": "suboptimal"},
    ],
    "testing": [
        {"action": "Define test cases covering edge cases", "quality": "good"},
        {"action": "Set up test fixtures and mocks", "quality": "good"},
        {"action": "Execute tests and verify results", "quality": "good"},
        {"action": "Analyze test coverage gaps", "quality": "good"},
        {"action": "Write tests that always pass", "quality": "error"},
        {"action": "Test only happy path scenarios", "quality": "suboptimal"},
    ],
    "data_analysis": [
        {"action": "Load and inspect dataset structure", "quality": "good"},
        {"action": "Clean and preprocess data", "quality": "good"},
        {"action": "Perform statistical analysis", "quality": "good"},
        {"action": "Visualize patterns and trends", "quality": "good"},
        {"action": "Draw conclusions from insufficient data", "quality": "error"},
        {"action": "Ignore data quality issues", "quality": "suboptimal"},
    ],
    "documentation": [
        {"action": "Identify target audience and use cases", "quality": "good"},
        {"action": "Document API interfaces and parameters", "quality": "good"},
        {"action": "Provide code examples and tutorials", "quality": "good"},
        {"action": "Keep documentation synchronized with code", "quality": "good"},
        {"action": "Document outdated or incorrect information", "quality": "error"},
        {"action": "Write documentation that assumes prior knowledge", "quality": "suboptimal"},
    ],
    "design": [
        {"action": "Gather user requirements and constraints", "quality": "good"},
        {"action": "Create wireframes or mockups", "quality": "good"},
        {"action": "Design user flows and interactions", "quality": "good"},
        {"action": "Iterate based on feedback", "quality": "good"},
        {"action": "Design without considering user needs", "quality": "error"},
        {"action": "Create overly complex interfaces", "quality": "suboptimal"},
    ],
    "automation": [
        {"action": "Identify repetitive manual tasks", "quality": "good"},
        {"action": "Design automation workflow", "quality": "good"},
        {"action": "Implement automation scripts", "quality": "good"},
        {"action": "Add error handling and monitoring", "quality": "good"},
        {"action": "Automate without understanding the process", "quality": "error"},
        {"action": "Create brittle automation that breaks easily", "quality": "suboptimal"},
    ],
    "integration": [
        {"action": "Analyze API specifications and requirements", "quality": "good"},
        {"action": "Handle authentication and authorization", "quality": "good"},
        {"action": "Implement data transformation layer", "quality": "good"},
        {"action": "Test integration with error scenarios", "quality": "good"},
        {"action": "Integrate without error handling", "quality": "error"},
        {"action": "Assume APIs never change or fail", "quality": "suboptimal"},
    ],
    "optimization": [
        {"action": "Profile and identify bottlenecks", "quality": "good"},
        {"action": "Analyze algorithmic complexity", "quality": "good"},
        {"action": "Implement performance improvements", "quality": "good"},
        {"action": "Measure and validate improvements", "quality": "good"},
        {"action": "Optimize without measuring impact", "quality": "error"},
        {"action": "Prematurely optimize non-critical paths", "quality": "suboptimal"},
    ],
    "security": [
        {"action": "Identify potential vulnerabilities", "quality": "good"},
        {"action": "Implement secure authentication", "quality": "good"},
        {"action": "Sanitize user inputs", "quality": "good"},
        {"action": "Follow security best practices", "quality": "good"},
        {"action": "Expose sensitive data in responses", "quality": "error"},
        {"action": "Use weak encryption or hashing", "quality": "suboptimal"},
    ],
    "deployment": [
        {"action": "Prepare build configuration", "quality": "good"},
        {"action": "Set up CI/CD pipeline", "quality": "good"},
        {"action": "Configure environment variables", "quality": "good"},
        {"action": "Test deployment in staging environment", "quality": "good"},
        {"action": "Deploy without testing", "quality": "error"},
        {"action": "Skip rollback procedures", "quality": "suboptimal"},
    ],
    "learning": [
        {"action": "Break down complex topic into concepts", "quality": "good"},
        {"action": "Find relevant learning resources", "quality": "good"},
        {"action": "Practice with examples and exercises", "quality": "good"},
        {"action": "Apply knowledge to solve problems", "quality": "good"},
        {"action": "Memorize without understanding", "quality": "error"},
        {"action": "Skip foundational concepts", "quality": "suboptimal"},
    ],
    "creative": [
        {"action": "Brainstorm multiple ideas and approaches", "quality": "good"},
        {"action": "Combine different concepts innovatively", "quality": "good"},
        {"action": "Iterate and refine creative output", "quality": "good"},
        {"action": "Seek feedback and incorporate it", "quality": "good"},
        {"action": "Stick rigidly to first idea", "quality": "error"},
        {"action": "Create without considering constraints", "quality": "suboptimal"},
    ],
    "communication": [
        {"action": "Clarify requirements and expectations", "quality": "good"},
        {"action": "Present information clearly and concisely", "quality": "good"},
        {"action": "Adapt message to audience", "quality": "good"},
        {"action": "Solicit and incorporate feedback", "quality": "good"},
        {"action": "Use jargon without explanation", "quality": "error"},
        {"action": "Provide too much or too little detail", "quality": "suboptimal"},
    ],
    "data_processing": [
        {"action": "Parse and validate input data format", "quality": "good"},
        {"action": "Transform data to required structure", "quality": "good"},
        {"action": "Apply business rules and filters", "quality": "good"},
        {"action": "Handle missing or invalid data gracefully", "quality": "good"},
        {"action": "Process data without validation", "quality": "error"},
        {"action": "Lose data during transformation", "quality": "suboptimal"},
    ],
    "api_development": [
        {"action": "Design RESTful endpoint structure", "quality": "good"},
        {"action": "Implement request validation", "quality": "good"},
        {"action": "Handle errors with appropriate status codes", "quality": "good"},
        {"action": "Document API with OpenAPI/Swagger", "quality": "good"},
        {"action": "Expose internal implementation details", "quality": "error"},
        {"action": "Return inconsistent response formats", "quality": "suboptimal"},
    ],
    "database": [
        {"action": "Design normalized schema", "quality": "good"},
        {"action": "Write efficient queries with indexes", "quality": "good"},
        {"action": "Handle transactions and concurrency", "quality": "good"},
        {"action": "Implement data migration scripts", "quality": "good"},
        {"action": "Create queries vulnerable to SQL injection", "quality": "error"},
        {"action": "Query without considering performance", "quality": "suboptimal"},
    ],
    "ui_development": [
        {"action": "Design responsive layout", "quality": "good"},
        {"action": "Implement accessible components", "quality": "good"},
        {"action": "Handle user interactions and state", "quality": "good"},
        {"action": "Optimize for performance and loading", "quality": "good"},
        {"action": "Create inaccessible interfaces", "quality": "error"},
        {"action": "Build without mobile consideration", "quality": "suboptimal"},
    ],
}

# -----------------------------------------------------------
# 2. PROMPT ANALYSIS → CHOOSE ACTION CATEGORY
# -----------------------------------------------------------

def categorize_prompt(prompt: str) -> str:
    """
    Categorize user prompts into task types for agentic task creation.
    Uses keyword matching, pattern recognition, and priority-based scoring.
    """
    p = prompt.lower()
    
    # Define category keywords with priority (more specific first)
    category_patterns = {
        "debugging": [
            "debug", "fix bug", "error", "crash", "broken", "not working", "issue",
            "troubleshoot", "diagnose", "trace", "stack trace", "exception",
            "fix error", "resolve issue", "bug fix", "debugging", "why isn't",
            "doesn't work", "won't work", "failing", "failure"
        ],
        "testing": [
            "test", "unit test", "integration test", "test case", "test suite",
            "write tests", "testing", "test coverage", "assert", "mock",
            "test data", "test scenario", "qa", "quality assurance"
        ],
        "refactoring": [
            "refactor", "clean up", "improve code", "restructure", "reorganize",
            "code quality", "technical debt", "code smell", "simplify code",
            "make it better", "optimize structure", "improve readability"
        ],
        "coding": [
            "write code", "implement", "create function", "build", "develop",
            "program", "code", "script", "algorithm", "function", "class",
            "module", "feature", "add functionality", "make a", "create a",
            "build a", "develop a", "programming", "software", "application"
        ],
        "data_analysis": [
            "analyze data", "data analysis", "dataset", "statistics", "statistical",
            "data visualization", "plot", "chart", "graph", "correlation",
            "trend", "pattern", "insight", "data science", "pandas", "numpy",
            "dataframe", "csv", "excel", "spreadsheet"
        ],
        "data_processing": [
            "process data", "transform data", "parse", "extract", "convert",
            "format data", "clean data", "filter data", "aggregate", "merge",
            "data pipeline", "etl", "data transformation"
        ],
        "database": [
            "database", "sql", "query", "table", "schema", "migration",
            "orm", "model", "entity", "relationship", "join", "index",
            "postgres", "mysql", "mongodb", "sqlite", "db"
        ],
        "api_development": [
            "api", "endpoint", "rest", "graphql", "http", "request", "response",
            "route", "controller", "service", "microservice", "webhook",
            "authentication", "authorization", "token", "jwt"
        ],
        "integration": [
            "integrate", "connect", "api integration", "third party", "external",
            "webhook", "sdk", "library integration", "service integration",
            "connect to", "link with", "sync with"
        ],
        "ui_development": [
            "ui", "user interface", "frontend", "react", "vue", "angular",
            "component", "button", "form", "page", "layout", "design ui",
            "user experience", "ux", "interface", "gui", "web page"
        ],
        "design": [
            "design", "mockup", "wireframe", "prototype", "ui design",
            "user experience design", "ux design", "visual design", "layout",
            "design system", "style guide", "branding"
        ],
        "documentation": [
            "document", "documentation", "readme", "api docs", "comment",
            "explain code", "how to use", "tutorial", "guide", "manual",
            "docstring", "javadoc", "code comments"
        ],
        "deployment": [
            "deploy", "deployment", "ci/cd", "pipeline", "docker", "kubernetes",
            "aws", "azure", "gcp", "cloud", "server", "production", "staging",
            "release", "build", "package", "container"
        ],
        "security": [
            "security", "encrypt", "decrypt", "hash", "password", "authentication",
            "authorization", "vulnerability", "secure", "ssl", "tls", "oauth",
            "xss", "csrf", "sql injection", "sanitize", "validate input"
        ],
        "optimization": [
            "optimize", "performance", "speed", "fast", "efficient", "bottleneck",
            "profiling", "benchmark", "improve performance", "make faster",
            "optimization", "cache", "lazy loading"
        ],
        "automation": [
            "automate", "automation", "script", "scheduled", "cron", "task",
            "workflow", "pipeline", "batch", "automated", "auto", "scheduler"
        ],
        "math": [
            "calculate", "equation", "compute", "math", "mathematical", "formula",
            "solve", "derivative", "integral", "statistics", "probability",
            "algebra", "calculus", "geometry", "number", "numeric"
        ],
        "analysis": [
            "analyze", "analysis", "examine", "evaluate", "assess", "review",
            "why", "how", "reasoning", "compare", "contrast", "investigate",
            "understand", "break down", "decompose", "study"
        ],
        "research": [
            "research", "search", "find", "look up", "information", "retrieve",
            "gather", "collect", "investigate", "explore", "discover",
            "learn about", "what is", "tell me about"
        ],
        "planning": [
            "plan", "strategy", "steps", "approach", "roadmap", "milestone",
            "timeline", "schedule", "how to", "method", "process", "workflow"
        ],
        "writing": [
            "write", "blog", "essay", "article", "content", "draft", "compose",
            "explain", "describe", "summarize", "documentation", "text",
            "prose", "narrative", "story"
        ],
        "learning": [
            "learn", "tutorial", "teach", "understand", "explain", "how does",
            "what is", "concept", "study", "course", "lesson", "education"
        ],
        "creative": [
            "create", "design", "brainstorm", "idea", "innovate", "creative",
            "artistic", "imagine", "invent", "original", "unique", "novel"
        ],
        "communication": [
            "present", "presentation", "communicate", "explain to", "tell",
            "share", "discuss", "meeting", "email", "message", "report"
        ],
    }
    
    # Score each category based on keyword matches
    scores = {}
    for category, keywords in category_patterns.items():
        score = 0
        for keyword in keywords:
            if keyword in p:
                # Longer keywords get higher weight
                score += len(keyword) * 2
                # Exact word matches get bonus
                if f" {keyword} " in f" {p} " or p.startswith(keyword) or p.endswith(keyword):
                    score += 10
        if score > 0:
            scores[category] = score
    
    # Return highest scoring category
    if scores:
        return max(scores, key=scores.get)
    
    # Fallback: check for common patterns
    if re.search(r'\b(how|what|why|when|where)\b', p):
        return "analysis"
    if re.search(r'\b(make|create|build|generate)\b', p):
        return "coding"
    if re.search(r'\b(fix|solve|resolve|help)\b', p):
        return "debugging"
    
    # Default fallback
    return "analysis"

# -----------------------------------------------------------
# 3. RUN SIMULATION — STEPS ARE NOW PROMPT-AWARE + QUALITY-BASED
# -----------------------------------------------------------

def determine_step_count(prompt: str, category: str) -> int:
    """
    Determine the number of steps needed based on prompt complexity and category.
    Returns a variable length between 3 and 8 steps.
    """
    # Base step counts by category (some tasks naturally need more steps)
    category_base_steps = {
        "debugging": 6,
        "testing": 5,
        "refactoring": 7,
        "coding": 6,
        "data_analysis": 5,
        "data_processing": 4,
        "database": 5,
        "api_development": 6,
        "integration": 7,
        "ui_development": 6,
        "design": 5,
        "documentation": 4,
        "deployment": 7,
        "security": 6,
        "optimization": 5,
        "automation": 5,
        "math": 4,
        "analysis": 5,
        "research": 6,
        "planning": 5,
        "writing": 4,
        "learning": 5,
        "creative": 5,
        "communication": 4,
    }
    
    # Start with category-based step count
    base_steps = category_base_steps.get(category, 5)
    
    # Adjust based on prompt length (longer prompts might need more steps)
    prompt_length_factor = len(prompt.split())
    if prompt_length_factor > 20:
        base_steps += 1
    elif prompt_length_factor > 10:
        base_steps += 0
    else:
        base_steps -= 1
    
    # Adjust based on complexity indicators
    complexity_keywords = ["complex", "multiple", "several", "various", "comprehensive", 
                          "detailed", "thorough", "extensive", "advanced"]
    if any(kw in prompt.lower() for kw in complexity_keywords):
        base_steps += 1
    
    # Ensure step count is within reasonable bounds (3-8)
    step_count = max(3, min(8, base_steps))
    
    return step_count

def generate_run(prompt, run_id):
    category = categorize_prompt(prompt)
    # Fallback to analysis if category not found
    pool = ACTIONS.get(category, ACTIONS["analysis"])

    # Determine variable step count based on prompt and category
    base_step_count = determine_step_count(prompt, category)
    
    # Add slight variation based on run_id to simulate different agent paths
    # This makes different runs of the same prompt have slightly different lengths
    variation = (run_id % 3) - 1  # -1, 0, or 1
    step_count = max(3, min(8, base_step_count + variation))
    
    # Construct variable-length steps (use available actions, cycling if needed)
    steps = []
    for i in range(step_count):
        # Cycle through available actions if we need more steps than available
        action_obj = pool[i % len(pool)]
        steps.append({
            "step_index": i,
            "action": action_obj["action"],
            "observation": f"Performed '{action_obj['action']}' in context of prompt: {prompt[:60]}...",
            "quality": action_obj["quality"],
            "is_error": action_obj["quality"] == "error",
            "is_suboptimal": action_obj["quality"] == "suboptimal"
        })

    return {
        "run_id": run_id,
        "category": category,  # Include category in response for debugging
        "steps": steps
    }

# -----------------------------------------------------------
# 4. ERROR AGGREGATION
# -----------------------------------------------------------

def derive_common_errors(simulations):
    error_counts = {}

    for sim in simulations:
        for step in sim["steps"]:
            if step["quality"] == "error":
                key = step["action"]
                error_counts[key] = error_counts.get(key, 0) + 1

    return [{"action": a, "count": c} for a, c in error_counts.items()]

# -----------------------------------------------------------
# 5. OPTIMAL PATH = all steps with quality = "good"
# -----------------------------------------------------------

def compute_optimal_path(simulations):
    path = []

    for step_idx in range(len(simulations[0]["steps"])):
        actions = []
        for sim in simulations:
            step = sim["steps"][step_idx]
            if step["quality"] == "good":
                actions.append(step["action"])

        if actions:
            # most frequent good action
            best = max(set(actions), key=actions.count)
            path.append(best)
        else:
            path.append("No valid step")

    return path

# -----------------------------------------------------------
# 6. API ENDPOINTS
# -----------------------------------------------------------

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True)
    prompt = data.get("prompt")
    num = data.get("num_simulations", 10)

    if not prompt:
        return jsonify({"error": "Prompt required"}), 400

    simulations = [generate_run(prompt, i) for i in range(num)]
    errors = derive_common_errors(simulations)
    optimal = compute_optimal_path(simulations)

    return jsonify({
        "prompt": prompt,
        "num_simulations": num,
        "simulations": simulations,
        "common_errors": errors,
        "optimal_path": optimal
    })


@app.route("/")
def root():
    return jsonify({"status": "Replay Lab backend running with prompt-aware actions"})


# -----------------------------------------------------------
# 7. HELPER FUNCTIONS FOR TESTING/DEBUGGING
# -----------------------------------------------------------

def simulate_prompt(prompt: str, num_simulations: int = 10):
    """
    Simulate a prompt and return the full API response structure.
    
    Args:
        prompt: The user prompt to simulate
        num_simulations: Number of simulation runs (default: 10)
    
    Returns:
        Dictionary matching the API response format
    """
    simulations = [generate_run(prompt, i) for i in range(num_simulations)]
    errors = derive_common_errors(simulations)
    optimal = compute_optimal_path(simulations)
    
    return {
        "prompt": prompt,
        "num_simulations": num_simulations,
        "simulations": simulations,
        "common_errors": errors,
        "optimal_path": optimal
    }

def print_prompt_output(prompt: str, num_simulations: int = 10):
    """
    Print the full JSON output for a given prompt.
    
    Args:
        prompt: The user prompt to simulate
        num_simulations: Number of simulation runs (default: 10)
    """
    result = simulate_prompt(prompt, num_simulations)
    
    print("=" * 80)
    print(f"PROMPT: {prompt}")
    print(f"CATEGORY: {result['simulations'][0]['category']}")
    print(f"NUMBER OF SIMULATIONS: {num_simulations}")
    print("=" * 80)
    print("\nFULL API RESPONSE (JSON):")
    print(json.dumps(result, indent=2))
    print("=" * 80)

def print_prompt_summary(prompt: str, num_simulations: int = 10):
    """
    Print a summary of the prompt simulation (more readable format).
    
    Args:
        prompt: The user prompt to simulate
        num_simulations: Number of simulation runs (default: 10)
    """
    result = simulate_prompt(prompt, num_simulations)
    
    print("=" * 80)
    print(f"PROMPT: {prompt}")
    print(f"CATEGORY: {result['simulations'][0]['category']}")
    print(f"NUMBER OF SIMULATIONS: {num_simulations}")
    print("=" * 80)
    
    # Show step count variation
    step_counts = [len(sim['steps']) for sim in result['simulations']]
    print(f"\nStep Counts per Simulation: {step_counts}")
    print(f"Average Steps: {sum(step_counts) / len(step_counts):.1f}")
    print(f"Min Steps: {min(step_counts)}, Max Steps: {max(step_counts)}")
    
    # Show all simulation details
    print(f"\n{'=' * 80}")
    print("ALL SIMULATIONS:")
    print("=" * 80)
    
    for sim in result['simulations']:
        print(f"\nSimulation {sim['run_id']}:")
        print(f"  Category: {sim['category']}")
        print(f"  Number of Steps: {len(sim['steps'])}")
        print("  Steps:")
        for step in sim['steps']:
            quality_indicator = "✓" if step['quality'] == "good" else "✗" if step['quality'] == "error" else "~"
            print(f"    [{step['step_index']}] {quality_indicator} {step['action']} (quality: {step['quality']})")
        print("-" * 80)
    
    # Show common errors
    if result['common_errors']:
        print(f"\nCommon Errors ({len(result['common_errors'])}):")
        for error in result['common_errors']:
            print(f"  - {error['action']}: {error['count']} occurrences")
    else:
        print("\nNo common errors found.")
    
    # Show optimal path
    print(f"\nOptimal Path ({len(result['optimal_path'])} steps):")
    for i, action in enumerate(result['optimal_path']):
        print(f"  [{i}] {action}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Example usage (uncomment to test):
    # print_prompt_output("write a function to sort a list", num_simulations=5)
    # print_prompt_summary("debug this error in my code", num_simulations=5)
    # print_prompt_summary("Develop an algorithm to solve the travelling salesman problem", num_simulations=5)
    app.run(debug=True)
