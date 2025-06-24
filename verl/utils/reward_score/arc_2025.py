import re
import numpy as np
import ast


def try_parse_numpy_array(s):
    try:
        parsed = ast.literal_eval(str(s))
        if isinstance(parsed, (list, tuple)):
            return np.array(parsed)
        if isinstance(parsed, np.ndarray):
            return parsed
        return None
    
    except Exception as e:
        return None


def extract_solution(solution_str):
    # match = re.search(r'\\boxed\{(.*?)\}', solution_str)
    # if match:
    #     return match.group(1)
    # return None
    parts = solution_str.split("### Output", 1)
    return parts[1].strip() if len(parts) > 1 else None


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0):

    answer = extract_solution(solution_str)
    answer = try_parse_numpy_array(answer)
    target = try_parse_numpy_array(ground_truth)

    # print("")
    # print("*******************************************************")
    # print("*******************************************************")
    # print("*******************************************************")
    # print("*******************************************************")
    # print("*******************************************************")
    # print("")

    # print("\nSOLUTION_STR:\n")
    # print(solution_str)

    # print("\nGROUND_TRUTH:\n")
    # print(ground_truth)

    # print("\nANSWER:\n")
    # print(answer)

    # print("\nTARGET:\n")
    # print(target)


    try:
        reward = sum((answer == target).flatten()) / np.prod(target.shape)
    except Exception as e:
        reward = 0

    print(f"REWARD: {reward}")

    return reward


    # if answer is None:
    #     return 0
    # else:
    #     if answer == ground_truth:
    #         return score
    #     else:
    #         return format_score
