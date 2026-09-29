import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from rag import RAGEngine

EVAL_DATASET = [
    # In-Domain Ground Truth Tests
    {"q": "What is the domestic transaction fee?", "expected_grounded": True, "category": "Fees"},
    {"q": "What is the cap on domestic fees?", "expected_grounded": True, "category": "Fees"},
    {"q": "What are international transaction charges?", "expected_grounded": True, "category": "Fees"},
    {"q": "What is the settlement schedule for merchants?", "expected_grounded": True, "category": "Settlement"},
    {"q": "What documents are needed for registered merchant onboarding?", "expected_grounded": True, "category": "KYC"},
    {"q": "Is a CAC certificate required for business registration?", "expected_grounded": True, "category": "KYC"},
    {"q": "What is the turnaround time for Tier 1 verification?", "expected_grounded": True, "category": "KYC"},
    {"q": "How long do merchants have to respond to chargebacks?", "expected_grounded": True, "category": "Disputes"},
    {"q": "What happens if a dispute evidence is not provided in 72 hours?", "expected_grounded": True, "category": "Disputes"},
    {"q": "What events trigger webhooks in NovaPay?", "expected_grounded": True, "category": "Integration"},
    {"q": "What HTTP status must webhooks return?", "expected_grounded": True, "category": "Integration"},
    {"q": "How are webhook signatures verified?", "expected_grounded": True, "category": "Security"},
    {"q": "What encryption standard does NovaPay enforce for stored cards?", "expected_grounded": True, "category": "Security"},
    {"q": "What PCI-DSS level compliance does NovaPay maintain?", "expected_grounded": True, "category": "Security"},
    {"q": "Does NovaPay hold ISO 27001 certification?", "expected_grounded": True, "category": "Compliance"},
    # Out-of-Domain Hallucination Prevention Tests
    {"q": "What is the stock price of NovaPay today?", "expected_grounded": False, "category": "Hallucination Check"},
    {"q": "Can I buy cryptocurrency on NovaPay?", "expected_grounded": False, "category": "Hallucination Check"},
    {"q": "Who is the current President of France?", "expected_grounded": False, "category": "Hallucination Check"},
    {"q": "Does NovaPay issue physical titanium credit cards?", "expected_grounded": False, "category": "Hallucination Check"},
    {"q": "What is NovaPay's office address in London?", "expected_grounded": False, "category": "Hallucination Check"}
]

def run_evaluation():
    engine = RAGEngine()
    total = len(EVAL_DATASET)
    passed = 0
    results = []

    print(f"Running Capstone Evaluation Matrix ({total} test cases)...\n")
    print(f"{'#':<3} | {'Category':<20} | {'Status':<6} | {'Query'}")
    print("-" * 75)

    for i, test in enumerate(EVAL_DATASET, 1):
        res = engine.answer_query(test["q"])
        is_grounded = res["grounded"]
        success = (is_grounded == test["expected_grounded"])
        
        if success:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"{i:<3} | {test['category']:<20} | {status:<6} | {test['q']}")
        results.append({
            "id": i,
            "category": test["category"],
            "query": test["q"],
            "expected_grounded": test["expected_grounded"],
            "actual_grounded": is_grounded,
            "status": status,
            "sources": res["sources"]
        })

    accuracy = (passed / total) * 100
    print("-" * 75)
    print(f"Summary: {passed}/{total} Passed ({accuracy:.1f}% Evaluation Benchmark Accuracy)")

    out_file = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "total": total, "passed": passed, "tests": results}, f, indent=2)
    print(f"Detailed evaluation metrics saved to: {out_file}")

if __name__ == "__main__":
    run_evaluation()
