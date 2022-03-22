"""
This wrapper is only if you don't want to use libLogHelper
"""

from libLogHelper.build import log_helper

NOT_GOLDEN_GENERATION = True


def start_iteration():
    if NOT_GOLDEN_GENERATION:
        log_helper.start_iteration()


def end_iteration():
    if NOT_GOLDEN_GENERATION:
        log_helper.end_iteration()


#
# def start_log_file(bench_name: str, header: str):
#     if NOT_GOLDEN_GENERATION:
#         log_helper.start_log_file(benchmark_name=bench_name, test_info=header)
def start_setup_log_file(framework_name: str, args_conf: str, model_name: str, max_errors_per_iteration: int,
                         generate: bool):
    global NOT_GOLDEN_GENERATION
    NOT_GOLDEN_GENERATION = not generate
    if NOT_GOLDEN_GENERATION:
        dnn_log_header = f"framework:{framework_name} {args_conf}"
        bench_name = f"{framework_name}-{model_name}"
        log_helper.start_log_file(bench_name, dnn_log_header)
        log_helper.set_max_errors_iter(max_errors_per_iteration)


def end_log_file():
    if NOT_GOLDEN_GENERATION:
        log_helper.end_log_file()


def log_info_detail(info_detail: str):
    if NOT_GOLDEN_GENERATION:
        log_helper.log_info_detail(info_detail)


def log_error_detail(error_detail: str):
    if NOT_GOLDEN_GENERATION:
        log_helper.log_error_detail(error_detail)


def log_error_count(error_count: int):
    if NOT_GOLDEN_GENERATION:
        log_helper.log_error_count(error_count)


def log_info_count(info_count: int):
    if NOT_GOLDEN_GENERATION:
        log_helper.log_info_count(info_count)


def set_max_errors_iter(max_errors):
    if NOT_GOLDEN_GENERATION:
        return log_helper.set_max_errors_iter(max_errors)


def set_max_infos_iter(max_infos):
    if NOT_GOLDEN_GENERATION:
        return log_helper.set_max_infos_iter(max_infos)
