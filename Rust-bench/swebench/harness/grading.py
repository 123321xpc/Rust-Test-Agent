from pathlib import Path
from typing import Any

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    DetailedResolvedStatus,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.test_spec import TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, parse_log_cargo


# MARK: Utility functions
def test_passed(case: str, sm: dict[str, str]) -> bool:
    return case in sm and sm[case] == TestStatus.PASSED.value


def test_failed(case: str, sm: dict[str, str]) -> bool:
    return case not in sm or any(
        sm[case] == status for status in [TestStatus.FAILED.value, TestStatus.ERROR.value]
    )


# MARK: Evaluation report functions
def get_logs_eval(log_fp: str, test_output_fp: str) -> tuple[dict[str, str], bool]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        log_fp (str): path to log file
        test_output_fp: path to test output file
    Returns:
        bool: whether the patch applied successfully
        dict: status map
    
    TODO(john-b-yang): Check this is working properly...
    """
    # Convert e.g. "logs/scikit-learn__scikit-learn-12421/test_output.txt" to "scikit-learn/scikit-learn"
    sample_id = str(Path(log_fp).parent.stem)  # e.g. scikit-learn__scikit-learn-12421
    repo = "-".join(sample_id.replace("__", "/").split("-")[:-1])  # e.g. scikit-learn/scikit-learn
    # log_parser = MAP_REPO_TO_PARSER[repo]
    log_parser = MAP_REPO_TO_PARSER.get(repo, parse_log_cargo)

    with open(log_fp) as f, open(test_output_fp) as f2:
        content = f.read()
        # TODO fix constant here
        if (
            any(
                [
                    x in content
                    for x in [
                        APPLY_PATCH_FAIL,
                        RESET_FAILED,
                        TESTS_ERROR,
                        TESTS_TIMEOUT,
                        "Failed to reset task environment",
                    ]
                ]
            )
            or APPLY_PATCH_PASS not in content
        ):
            # Eval patch was not applied successfully
            return {}, False

        # if patch is applied sucessfully, get status map of evaluation results
        test_content = f2.read()
        return log_parser(test_content), True


def get_eval_tests_report(
    eval_sm: dict[str, str],
    gold_results: dict[str, str],
    calculate_to_fail: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change from gold results to eval results.

    Args:
        eval_sm (dict): evaluation status map
        gold_results (dict): gold results
        calculate_to_fail (bool): whether to calculate metrics for "x to fail" tests
    Returns:
        report (dict): report of metrics

    Metric Definitions (Gold Result Pair + Eval Result):
    - Fail-Pass (F2P) + P: Success (Resolution)
    - Pass-Pass (P2P) + P: Success (Maintenance)
    - Fail-Pass (F2P) + F: Failure
    - Pass-Pass (P2P) + F: Failure

    Miscellaneous Definitions
    - Fail-Fail (F2F) + F: Failure Maintenance
    - Pass-Fail (P2F) + F: Not considered
    - Fail-Fail (F2F) + P: Success (Extra Credit)
    - Pass-Fail (P2F) + P: Not considered
    """
    # Calculate resolution metrics
    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        if test_passed(test_case, eval_sm):
            # Assume silent success for now (test case not in eval_sm)
            f2p_success.append(test_case)
        elif test_failed(test_case, eval_sm):
            f2p_failure.append(test_case)

    # Calculate maintenance metrics
    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        if test_passed(test_case, eval_sm):
            p2p_success.append(test_case)
        elif test_failed(test_case, eval_sm):
            p2p_failure.append(test_case)

    results = {
        FAIL_TO_PASS: {
            "success": f2p_success,
            "failure": f2p_failure,
        },
        PASS_TO_PASS: {
            "success": p2p_success,
            "failure": p2p_failure,
        },
    }

    f2f_success = []
    f2f_failure = []
    p2f_success = []
    p2f_failure = []
    if calculate_to_fail:
        # Calculate "extra credit" metrics
        for test_case in gold_results[FAIL_TO_FAIL]:
            if test_passed(test_case, eval_sm):
                f2f_success.append(test_case)
            elif test_failed(test_case, eval_sm):
                f2f_failure.append(test_case)

        # Calculate not considered metrics
        for test_case in gold_results[PASS_TO_FAIL]:
            if test_passed(test_case, eval_sm):
                p2f_success.append(test_case)
            elif test_failed(test_case, eval_sm):
                p2f_failure.append(test_case)

    results.update(
        {
            FAIL_TO_FAIL: {
                "success": f2f_success,
                "failure": f2f_failure,
            },
            PASS_TO_FAIL: {
                "success": p2f_success,
                "failure": p2f_failure,
            },
        }
    )
    return results


def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute fail-to-pass metric. Accepts single report as argument.
    """
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]["success"]) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute pass-to-pass metric. Accepts single report as argument.
    """
    total = len(report[PASS_TO_PASS]["success"]) + len(report[PASS_TO_PASS]["failure"])
    if total == 0:
        # TODO: Don't factor in p2p metrics
        return 1
    return len(report[PASS_TO_PASS]["success"]) / total


def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    """
    Determine resolved status of an evaluation instance

    Criteria:
        - If fail-to-pass (Resolution) = 1 and pass-to-pass (Maintenance) = 1 -> FULL
        - If (fail-to-pass (Resolution) < 1 and > 0) and pass-to-pass (Maintenance) = 1 -> PARTIAL
        - Otherwise -> NO
    """
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    elif f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    else:
        return ResolvedStatus.NO.value
    

def classify_detailed_status(
    tests_report: dict[str, dict[str, Any]],
    patch_successfully_applied: bool = True,
    patch_exists: bool = True,
) -> str:
    """
    Categorize the outcome of an evaluation run into the 6 fine-grained classes
    defined in SWE-bench Appendix C.5 (Table 22):
    - Resolved: All F2P pass, All P2P pass
    - Breaking Resolved: All F2P pass, but some or all P2P fail
    - Partially Resolved: Some F2P pass, All P2P pass
    - Work in Progress: Some F2P pass, but some or all P2P fail
    - No-Op: No F2P pass, All P2P pass
    - Regression: No F2P pass, and some or all P2P fail
    """
    if not patch_exists:
        return DetailedResolvedStatus.EMPTY_PATCH.value
    if not patch_successfully_applied:
        return DetailedResolvedStatus.PATCH_APPLY_FAIL.value

    f2p = tests_report.get(FAIL_TO_PASS, {})
    p2p = tests_report.get(PASS_TO_PASS, {})

    f2p_succ = len(f2p.get("success", []))
    f2p_fail = len(f2p.get("failure", []))
    p2p_succ = len(p2p.get("success", []))
    p2p_fail = len(p2p.get("failure", []))

    f2p_total = f2p_succ + f2p_fail
    p2p_total = p2p_succ + p2p_fail

    if f2p_total == 0 or f2p_fail == 0:
        f2p_rate = "All"
    elif f2p_succ > 0:
        f2p_rate = "Partial"
    else:
        f2p_rate = "None"

    if p2p_total == 0 or p2p_fail == 0:
        p2p_rate = "All"
    elif p2p_succ > 0:
        p2p_rate = "Partial"
    else:
        p2p_rate = "None"

    matrix = {
        ("All", "All"): DetailedResolvedStatus.RESOLVED.value,
        ("All", "Partial"): DetailedResolvedStatus.BREAKING_RESOLVED.value,
        ("All", "None"): DetailedResolvedStatus.BREAKING_RESOLVED.value,
        ("Partial", "All"): DetailedResolvedStatus.PARTIALLY_RESOLVED.value,
        ("Partial", "Partial"): DetailedResolvedStatus.WORK_IN_PROGRESS.value,
        ("Partial", "None"): DetailedResolvedStatus.WORK_IN_PROGRESS.value,
        ("None", "All"): DetailedResolvedStatus.NO_OP.value,
        ("None", "Partial"): DetailedResolvedStatus.REGRESSION.value,
        ("None", "None"): DetailedResolvedStatus.REGRESSION.value,
    }
    return matrix.get((f2p_rate, p2p_rate), DetailedResolvedStatus.NO_OP.value)


def get_eval_report(
    test_spec: TestSpec,
    prediction: dict[str, str],
    log_path: str,
    test_output_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    report_map = {}

    instance_id = prediction[KEY_INSTANCE_ID]
    if instance_id not in report_map:
        report_map[instance_id] = {
            "patch_is_None": False,
            "patch_exists": False,
            "patch_successfully_applied": False,
            "resolved": False,
            "detailed_status": DetailedResolvedStatus.NO_OP.value,
        }

    if prediction["model_patch"] is None:
        report_map[instance_id]["none"] = True
        report_map[instance_id]["detailed_status"] = DetailedResolvedStatus.EMPTY_PATCH.value
        return report_map
    report_map[instance_id]["patch_exists"] = True

    eval_sm, found = get_logs_eval(log_path, test_output_path)

    if not found:
        report_map[instance_id]["detailed_status"] = DetailedResolvedStatus.PATCH_APPLY_FAIL.value
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    report = get_eval_tests_report(eval_sm, eval_ref)
    detailed_status = classify_detailed_status(
        report,
        patch_successfully_applied=True,
        patch_exists=True,
    )
    if detailed_status == DetailedResolvedStatus.RESOLVED.value:
        report_map[instance_id]["resolved"] = True
    report_map[instance_id]["detailed_status"] = detailed_status

    if include_tests_status:
        report_map[instance_id]["tests_status"] = report  # type: ignore
    
    return report_map
