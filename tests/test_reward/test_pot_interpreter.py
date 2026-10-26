'Tests for the safe PoT interpreter (whitelist + canonicalization).\n\n\nPorted verbatim from the source project ``ragu_for_docsem``\n(``tests/test_interpreter.py``, 27 tests including the attacking programs).'

from __future__ import annotations

import pytest

from llm_training.reward.pot_interpreter import (
    ProgramError,
    execute_program,
    format_result,
)


class TestValidPrograms:
    def test_simple_arithmetic(self):
        assert execute_program("answer = 8400 - 300\nresult = answer / 900") == 9

    def test_augmented_assignment(self):
        assert execute_program("answer = 2\nanswer += 3\nanswer *= 4") == 20

    def test_functions(self):
        assert execute_program("answer = max(3, 7) + min(10, 2)") == 9
        assert execute_program("answer = round(8100 / 900)") == 9
        assert execute_program("answer = sum([1, 2, 3]) * abs(-2)") == 12

    def test_math_names(self):
        assert execute_program("answer = ceil(195 / 15)") == 13
        assert execute_program("answer = math.floor(7.9) + math.sqrt(16)") == 11

    def test_conditional(self):
        assert execute_program(
            "x = 5\nanswer = x if x > 3 else 0"
        ) == 5

    def test_precedence_and_floats(self):
        assert execute_program("answer = 0.6 + 36 * 0.5") == pytest.approx(18.6)

    def test_list_intermediate_values(self):
        assert execute_program("values = [1, 2, 3]\nanswer = sum(values)") == 6


class TestRejectedPrograms:
    def test_import(self):
        with pytest.raises(ProgramError):
            execute_program("import os")

    def test_dunder_import(self):
        with pytest.raises(ProgramError):
            execute_program("answer = __import__('os').getcwd()")

    def test_while_loop(self):
        with pytest.raises(ProgramError):
            execute_program("while True:\n    pass")

    def test_for_loop(self):
        with pytest.raises(ProgramError):
            execute_program("total = 0\nfor i in [1, 2]:\n    total += i")

    def test_attribute_access(self):
        with pytest.raises(ProgramError):
            execute_program("answer = (1).bit_length()")

    def test_disallowed_function(self):
        with pytest.raises(ProgramError):
            execute_program("answer = print(1)")

    def test_string_constant(self):
        with pytest.raises(ProgramError):
            execute_program("answer = 'abc'")

    def test_huge_power(self):
        with pytest.raises(ProgramError):
            execute_program("answer = 2 ** 10000")

    def test_overflow_value(self):
        with pytest.raises(ProgramError):
            execute_program("answer = 10 ** 14 * 10 ** 3")

    def test_division_by_zero(self):
        with pytest.raises(ProgramError):
            execute_program("answer = 1 / 0")

    def test_syntax_error(self):
        with pytest.raises(ProgramError):
            execute_program("answer = = 2")

    def test_empty(self):
        with pytest.raises(ProgramError):
            execute_program("   ")

    def test_undefined_variable(self):
        with pytest.raises(ProgramError):
            execute_program("answer = unknown + 1")

    def test_no_assignment(self):
        with pytest.raises(ProgramError):
            execute_program("1 + 2")

    def test_too_many_nodes(self):
        with pytest.raises(ProgramError):
            execute_program("answer = " + " + ".join(["1"] * 200))

    def test_lambda_or_def(self):
        with pytest.raises(ProgramError):
            execute_program("f = lambda x: x\nanswer = f(1)")

    def test_list_final_program(self):
        # Lists are legal intermediates but the FINAL value must be numeric
        # (the portal answer is a single number) — regression for the
        # ``TypeError: type list doesn't define __round__ method`` crash.
        
        
        with pytest.raises(ProgramError):
            execute_program("answer = [58]")
        with pytest.raises(ProgramError):
            execute_program("values = [98, 40]\nanswer = values")


class TestFormatResult:
    def test_integer(self):
        assert format_result(1067) == "1067"

    def test_float_near_integer(self):
        assert format_result(9.0000000001) == "9"

    def test_decimal(self):
        assert format_result(18.6) == "18.6"

    def test_negative(self):
        assert format_result(-3.0) == "-3"

    def test_long_decimal_trimmed(self):
        assert format_result(1 / 3) == "0.3333333333"

    def test_non_numeric_rejected(self):
        # Defense in depth for the ``execute_program`` guard: a non-numeric
        # value raises ProgramError, not the cryptic round() TypeError.
        
        with pytest.raises(ProgramError):
            format_result([1, 2])
        with pytest.raises(ProgramError):
            format_result(True)
