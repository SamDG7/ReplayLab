import json
import pytest
from app import app, categorize_prompt, ACTIONS  # import the Flask app and functions


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_root_endpoint(client):
    """Ensure the root endpoint responds correctly."""
    print("\n" + "="*80)
    print("TEST: test_root_endpoint")
    print("="*80)
    print("INPUT: GET /")
    
    res = client.get("/")
    assert res.status_code == 200
    data = res.get_json()
    
    print("OUTPUT:")
    print(json.dumps(data, indent=2))
    print("="*80)
    
    assert data["status"] == "Replay Lab backend running with prompt-aware actions"


def test_simulate_basic(client):
    """A simple test for the /simulate endpoint."""
    payload = {
        "prompt": "Test prompt for Replay Lab",
        "num_simulations": 5
    }

    print("\n" + "="*80)
    print("TEST: test_simulate_basic")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))

    res = client.post("/simulate", json=payload)
    assert res.status_code == 200

    data = res.get_json()

    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(f"Prompt: {data.get('prompt')}")
    print(f"Category: {data['simulations'][0].get('category')}")
    print(f"Number of Simulations: {data.get('num_simulations')}")
    print(f"Number of Steps per Simulation: {len(data['simulations'][0]['steps'])}")
    print("\nFirst Simulation Steps:")
    for step in data['simulations'][0]['steps']:
        print(f"  [{step['step_index']}] {step['action']} (quality: {step['quality']})")
    print(f"\nCommon Errors: {len(data.get('common_errors', []))}")
    print(f"Optimal Path Length: {len(data.get('optimal_path', []))}")
    print("\nFull Response:")
    print(json.dumps(data, indent=2))
    print("="*80)

    # Check top-level keys
    assert "prompt" in data
    assert "num_simulations" in data
    assert "simulations" in data
    assert "common_errors" in data
    assert "optimal_path" in data

    # Check simulation count
    assert len(data["simulations"]) == 5
    assert data["num_simulations"] == 5

    # Check that each run has required fields
    for sim in data["simulations"]:
        assert "run_id" in sim
        assert "category" in sim
        assert "steps" in sim
        assert len(sim["steps"]) >= 1
        assert isinstance(sim["category"], str)
        assert sim["category"] in ACTIONS


def test_simulate_missing_prompt(client):
    """Ensure backend rejects missing prompt."""
    payload = {}
    
    print("\n" + "="*80)
    print("TEST: test_simulate_missing_prompt")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))
    
    res = client.post("/simulate", json=payload)
    assert res.status_code == 400
    data = res.get_json()
    
    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(json.dumps(data, indent=2))
    print("="*80)
    
    assert "error" in data
    assert data["error"] == "Prompt required"


def test_simulate_default_num_simulations(client):
    """Test that default num_simulations is 10 when not provided."""
    payload = {
        "prompt": "Test prompt"
    }

    print("\n" + "="*80)
    print("TEST: test_simulate_default_num_simulations")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))

    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    
    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(f"Prompt: {data.get('prompt')}")
    print(f"Category: {data['simulations'][0].get('category')}")
    print(f"Number of Simulations (default): {data.get('num_simulations')}")
    print(f"Actual Simulations Count: {len(data['simulations'])}")
    print("="*80)
    
    assert data["num_simulations"] == 10
    assert len(data["simulations"]) == 10


def test_simulate_error_and_suboptimal_flags(client):
    """Verify runs include error + suboptimal fields."""
    payload = {
        "prompt": "Another test prompt",
        "num_simulations": 3
    }

    print("\n" + "="*80)
    print("TEST: test_simulate_error_and_suboptimal_flags")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))

    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    # Check steps include expected fields
    steps = data["simulations"][0]["steps"]
    step = steps[0]

    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(f"Prompt: {data.get('prompt')}")
    print(f"Category: {data['simulations'][0].get('category')}")
    print("\nFirst Step Details:")
    print(f"  Step Index: {step.get('step_index')}")
    print(f"  Action: {step.get('action')}")
    print(f"  Quality: {step.get('quality')}")
    print(f"  Is Error: {step.get('is_error')}")
    print(f"  Is Suboptimal: {step.get('is_suboptimal')}")
    print("\nAll Steps Quality Breakdown:")
    for i, s in enumerate(steps):
        print(f"  Step {i}: {s['quality']} (error: {s['is_error']}, suboptimal: {s['is_suboptimal']})")
    print("="*80)

    assert "step_index" in step
    assert "action" in step
    assert "observation" in step
    assert "quality" in step
    assert "is_error" in step
    assert "is_suboptimal" in step
    assert isinstance(step["is_error"], bool)
    assert isinstance(step["is_suboptimal"], bool)
    assert step["quality"] in ["good", "error", "suboptimal"]


def test_optimal_path_length(client):
    """Ensure optimal path is the same length as the longest simulation."""
    payload = {
        "prompt": "Check optimal path length",
        "num_simulations": 5
    }

    print("\n" + "="*80)
    print("TEST: test_optimal_path_length")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))

    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    sim_lengths = [len(sim["steps"]) for sim in data["simulations"]]
    max_len = max(sim_lengths)

    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(f"Prompt: {data.get('prompt')}")
    print(f"Category: {data['simulations'][0].get('category')}")
    print(f"Simulation Step Lengths: {sim_lengths}")
    print(f"Max Length: {max_len}")
    print(f"Optimal Path Length: {len(data['optimal_path'])}")
    print("\nOptimal Path:")
    for i, action in enumerate(data['optimal_path']):
        print(f"  [{i}] {action}")
    print("="*80)

    assert len(data["optimal_path"]) == max_len


def test_categorize_prompt_coding():
    """Test that coding-related prompts are categorized correctly."""
    test_cases = [
        "write a function to sort a list",
        "implement a sorting algorithm",
        "create a python script",
        "build a web application",
        "develop a feature"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_coding")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "coding"
        print()
    
    print("="*80)


def test_categorize_prompt_debugging():
    """Test that debugging-related prompts are categorized correctly."""
    test_cases = [
        "fix this bug",
        "debug error in my code",
        "troubleshoot why it's not working",
        "resolve this issue"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_debugging")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "debugging"
        print()
    
    print("="*80)


def test_categorize_prompt_testing():
    """Test that testing-related prompts are categorized correctly."""
    test_cases = [
        "write unit tests",
        "create test cases",
        "test coverage"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_testing")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "testing"
        print()
    
    print("="*80)


def test_categorize_prompt_data_analysis():
    """Test that data analysis prompts are categorized correctly."""
    test_cases = [
        "analyze this dataset",
        "data visualization",
        "statistical analysis"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_data_analysis")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "data_analysis"
        print()
    
    print("="*80)


def test_categorize_prompt_api_development():
    """Test that API development prompts are categorized correctly."""
    test_cases = [
        "create a REST API endpoint",
        "build an API",
        "design REST endpoints"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_api_development")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "api_development"
        print()
    
    print("="*80)


def test_categorize_prompt_documentation():
    """Test that documentation prompts are categorized correctly."""
    test_cases = [
        "write documentation",
        "create API docs",
        "document this code"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_documentation")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "documentation"
        print()
    
    print("="*80)


def test_categorize_prompt_database():
    """Test that database prompts are categorized correctly."""
    test_cases = [
        "write a SQL query",
        "database migration",
        "create database schema"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_database")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "database"
        print()
    
    print("="*80)


def test_categorize_prompt_math():
    """Test that math prompts are categorized correctly."""
    test_cases = [
        "calculate the derivative",
        "solve this equation",
        "mathematical formula"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_math")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "math"
        print()
    
    print("="*80)


def test_categorize_prompt_analysis():
    """Test that analysis prompts are categorized correctly."""
    test_cases = [
        "analyze this problem",
        "why does this happen",
        "compare these options"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_analysis")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "analysis"
        print()
    
    print("="*80)


def test_categorize_prompt_research():
    """Test that research prompts are categorized correctly."""
    test_cases = [
        "research this topic",
        "find information about",
        "search for"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_research")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "research"
        print()
    
    print("="*80)


def test_categorize_prompt_planning():
    """Test that planning prompts are categorized correctly."""
    test_cases = [
        "create a plan",
        "what are the steps",
        "strategy for"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_planning")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "planning"
        print()
    
    print("="*80)


def test_categorize_prompt_writing():
    """Test that writing prompts are categorized correctly."""
    test_cases = [
        "write a blog post",
        "compose an essay",
        "explain this concept"
    ]
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_writing")
    print("="*80)
    
    for prompt in test_cases:
        category = categorize_prompt(prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {category}")
        assert category == "writing"
        print()
    
    print("="*80)


def test_categorize_prompt_fallback():
    """Test that unknown prompts fall back to analysis."""
    test_prompt = "some random text without clear category"
    
    print("\n" + "="*80)
    print("TEST: test_categorize_prompt_fallback")
    print("="*80)
    print(f"INPUT:  {test_prompt}")
    
    result = categorize_prompt(test_prompt)
    
    print(f"OUTPUT: {result}")
    print(f"Valid Category: {result in ACTIONS}")
    print("="*80)
    
    assert isinstance(result, str)
    assert result in ACTIONS


def test_simulate_category_consistency(client):
    """Test that all simulations for the same prompt get the same category."""
    payload = {
        "prompt": "write a function to calculate fibonacci",
        "num_simulations": 5
    }

    print("\n" + "="*80)
    print("TEST: test_simulate_category_consistency")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))

    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    # All simulations should have the same category for the same prompt
    categories = [sim["category"] for sim in data["simulations"]]
    
    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(f"Prompt: {data.get('prompt')}")
    print(f"Categories across all simulations: {categories}")
    print(f"Unique Categories: {set(categories)}")
    print(f"All Same Category: {len(set(categories)) == 1}")
    print("="*80)

    assert len(set(categories)) == 1  # All should be the same
    assert categories[0] == "coding"  # Should be categorized as coding


def test_simulate_step_structure(client):
    """Test that steps have the correct structure."""
    payload = {
        "prompt": "test prompt",
        "num_simulations": 1
    }

    print("\n" + "="*80)
    print("TEST: test_simulate_step_structure")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))

    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    steps = data["simulations"][0]["steps"]
    
    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(f"Prompt: {data.get('prompt')}")
    print(f"Category: {data['simulations'][0].get('category')}")
    print(f"Number of Steps: {len(steps)}")
    print("\nStep Structure:")
    for i, step in enumerate(steps):
        print(f"\n  Step {i}:")
        print(f"    step_index: {step.get('step_index')}")
        print(f"    action: {step.get('action')}")
        print(f"    quality: {step.get('quality')}")
        print(f"    is_error: {step.get('is_error')}")
        print(f"    is_suboptimal: {step.get('is_suboptimal')}")
        print(f"    observation: {step.get('observation')[:80]}...")
    print("="*80)

    assert len(steps) == 5  # Should use first 5 actions

    for i, step in enumerate(steps):
        assert step["step_index"] == i
        assert "action" in step
        assert "observation" in step
        assert "quality" in step
        assert step["quality"] in ["good", "error", "suboptimal"]


def test_common_errors_structure(client):
    """Test that common_errors has the correct structure."""
    payload = {
        "prompt": "test prompt",
        "num_simulations": 10
    }

    print("\n" + "="*80)
    print("TEST: test_common_errors_structure")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))

    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(f"Prompt: {data.get('prompt')}")
    print(f"Category: {data['simulations'][0].get('category')}")
    print(f"Number of Common Errors: {len(data.get('common_errors', []))}")
    print("\nCommon Errors:")
    for error in data.get('common_errors', []):
        print(f"  - {error.get('action')}: {error.get('count')} occurrences")
    print("="*80)

    assert isinstance(data["common_errors"], list)
    for error in data["common_errors"]:
        assert "action" in error
        assert "count" in error
        assert isinstance(error["count"], int)
        assert error["count"] > 0


def test_optimal_path_structure(client):
    """Test that optimal_path has the correct structure."""
    payload = {
        "prompt": "test prompt",
        "num_simulations": 5
    }

    print("\n" + "="*80)
    print("TEST: test_optimal_path_structure")
    print("="*80)
    print("INPUT:")
    print(json.dumps(payload, indent=2))

    res = client.post("/simulate", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    print("\nOUTPUT:")
    print(f"Status Code: {res.status_code}")
    print(f"Prompt: {data.get('prompt')}")
    print(f"Category: {data['simulations'][0].get('category')}")
    print(f"Optimal Path Length: {len(data.get('optimal_path', []))}")
    print("\nOptimal Path:")
    for i, action in enumerate(data.get('optimal_path', [])):
        print(f"  [{i}] {action}")
    print("="*80)

    assert isinstance(data["optimal_path"], list)
    assert len(data["optimal_path"]) > 0
    # Each item should be a string (action description)
    for action in data["optimal_path"]:
        assert isinstance(action, str)
