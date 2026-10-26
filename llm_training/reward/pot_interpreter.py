'Safe PoT program interpreter (AST whitelist, no ``eval``/``exec``).\n\n\nPorted verbatim (semantics identical, limits identical) from the DocSem\ncompetition project ``ragu_for_docsem`` (``docsem_ragu/interpreter.py``,\n© an anonymous research team) — the battle-tested engine behind the champion\nProgram-of-Thought pipeline (27 tests there, including attacking programs).\nThe model writes a short program whose final value is the variable\n``answer``; execution is constrained by an AST whitelist (the PAL/PoT\nrecipe: the interpreter does the arithmetic, the model is responsible for\nthe problem setup). Allowed: numbers, arithmetic, assignments, conditional\nexpressions, tuples/lists, and the functions ``min``/``max``/``round``/\n``abs``/``sum``/``ceil``/``floor``/``sqrt`` (bare names or ``math.*``).\nEverything else — imports, attributes, loops, arbitrary calls — is rejected\nwithout execution; the program is never passed to ``eval``/``exec``.\n\nProgram-of-Thought'

from __future__ import annotations

import ast
import math

# : Hard limits (identical to the source project).
MAX_PROGRAM_LENGTH = 2000
MAX_NODES = 220
MAX_ABS_VALUE = 1e15
MAX_POW_EXPONENT = 64.0


class ProgramError(ValueError):
    'The program was rejected by the whitelist or failed at runtime.'


_ALLOWED_FUNCTIONS: dict[str, object] = {
    "min": min,
    "max": max,
    "round": round,
    "abs": abs,
    "sum": sum,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
}
_MATH_ATTRIBUTES = {"ceil", "floor", "sqrt"}

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_CMP_OPS = {
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _check_value(value: float | int) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramError("non-numeric value / нечисловое значение")
    if abs(value) > MAX_ABS_VALUE:
        raise ProgramError(f"magnitude {value} out of bounds / величина вне пределов")
    return value


class _Validator(ast.NodeVisitor):
    'Validates the program tree against the whitelist.'

    def __init__(self) -> None:
        self.nodes = 0

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Attribute):
            # Only math.ceil / math.floor / math.sqrt in a call position are
            # allowed; any other attribute access is rejected.
            # math.floor
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"
                    and node.attr in _MATH_ATTRIBUTES):
                raise ProgramError("attribute access not allowed / атрибут запрещён")
        elif not isinstance(node, (
            ast.Module, ast.Assign, ast.AugAssign, ast.Expr, ast.BinOp,
            ast.UnaryOp, ast.IfExp, ast.Compare, ast.Call, ast.Name,
            ast.Load, ast.Store, ast.Constant, ast.Tuple, ast.List,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
            ast.Pow, ast.USub, ast.UAdd,
            ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
            ast.expr_context, ast.operator, ast.cmpop,
        )):
            raise ProgramError(
                f"construct {type(node).__name__} not allowed / конструкция запрещена"
            )
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ProgramError("too many nodes / слишком много узлов")
        super().generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ProgramError(
                "only numeric constants allowed / допустимы только числа"
            )

    def visit_Call(self, node: ast.Call) -> None:
        if node.keywords:
            raise ProgramError(
                "keyword arguments not allowed / именованные аргументы запрещены"
            )
        function = node.func
        if isinstance(function, ast.Attribute):
            if not (isinstance(function.value, ast.Name)
                    and function.value.id == "math"
                    and function.attr in _MATH_ATTRIBUTES):
                raise ProgramError(
                    "attribute access not allowed / атрибут запрещён"
                )
        elif isinstance(function, ast.Name):
            if function.id not in _ALLOWED_FUNCTIONS:
                raise ProgramError(
                    f"function {function.id!r} not allowed / функция запрещена"
                )
        else:
            raise ProgramError("unexpected call / неожидаемый вызов")
        self.generic_visit(node)


def _eval(node: ast.AST, env: dict[str, float | int]) -> float | int:
    if isinstance(node, ast.Constant):
        return _check_value(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ProgramError(f"variable {node.id!r} not defined / переменная не определена")
        return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(item, env) for item in node.elts]
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow):
            exponent = abs(float(right))
            if exponent > MAX_POW_EXPONENT:
                raise ProgramError("exponent too large / слишком большая степень")
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ProgramError(
                "invalid binary operation / недопустимая бинарная операция"
            )
        try:
            return _check_value(op(left, right))
        except ZeroDivisionError as exc:
            raise ProgramError("division by zero / деление на ноль") from exc
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return _check_value(-operand)
        if isinstance(node.op, ast.UAdd):
            return _check_value(operand)
        raise ProgramError("invalid unary operation / недопустимая унарная операция")
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise ProgramError(
                    "invalid comparison / недопустимое сравнение"
                )
            right = _eval(comparator, env)
            if not op(left, right):
                return 0.0
            left = right
        return 1.0
    if isinstance(node, ast.IfExp):
        condition = _eval(node.test, env)
        branch = node.body if condition else node.orelse
        return _eval(branch, env)
    if isinstance(node, ast.Call):
        function = node.func
        if isinstance(function, ast.Attribute):
            name = f"math.{function.attr}"
            target = getattr(math, function.attr)
        else:
            name = function.id
            target = _ALLOWED_FUNCTIONS[name]
        args = [_eval(arg, env) for arg in node.args]
        try:
            result = target(*args)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
            raise ProgramError(f"{name} failed: {exc} / вызов не удался") from exc
        return _check_value(result)
    raise ProgramError(
        f"expression {type(node).__name__} not supported / выражение не поддерживается"
    )


def execute_program(program: str) -> float | int:
    "Execute the program and return its final value.\n\n\n    The final value is the last assignment (the prompt requires ending with\n    ``answer = ...``; if the model did not, the last assignment is the best\n    hypothesis of the final). The final value must be **numeric**:\n    tuples/lists are legal intermediates (``sum([1, 2])``) but a list-final\n    program has no canonical answer string (the portal answer is a single\n    number) and is rejected. A rejected or failed execution raises\n    :class:`ProgramError` — the caller falls back to the model's stated\n    answer."
    if not program or not program.strip():
        raise ProgramError("empty program / пустая программа")
    if len(program) > MAX_PROGRAM_LENGTH:
        raise ProgramError("program too long / программа слишком длинная")
    try:
        tree = ast.parse(program, mode="exec")
    except SyntaxError as exc:
        raise ProgramError(f"syntax error / синтаксическая ошибка: {exc.msg}") from exc

    _Validator().visit(tree)

    env: dict[str, float | int] = {}
    last_name: str | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                raise ProgramError(
                    "assignment only to a simple variable / присваивание только "
                    "в простую переменную"
                )
            name = targets[0].id
            env[name] = _eval(statement.value, env)
            last_name = name
        elif isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name):
                raise ProgramError(
                    "assignment only to a simple variable / присваивание только "
                    "в простую переменную"
                )
            name = statement.target.id
            if name not in env:
                raise ProgramError(
                    f"variable {name!r} not defined / переменная не определена"
                )
            op = _BIN_OPS.get(type(statement.op))
            if op is None:
                raise ProgramError(
                    "invalid assignment operation / недопустимая операция присваивания"
                )
            env[name] = _check_value(op(env[name], _eval(statement.value, env)))
            last_name = name
        elif isinstance(statement, ast.Expr):
            _eval(statement.value, env)
        else:
            raise ProgramError(
                f"statement {type(statement).__name__} not allowed / инструкция запрещена"
            )

    if last_name is not None:
        return _check_value(env[last_name])
    raise ProgramError("no assignments in program / в программе нет присваиваний")


def format_result(value: float | int) -> str:
    "Canonical value string: integer without a dot, decimal without tails.\n\n\n    Non-numeric values (e.g. a list-final program result reaching this\n    function despite the ``execute_program`` guard) are rejected with\n    :class:`ProgramError` — not the cryptic ``TypeError`` from ``round()`` —\n    so the caller's ``except ProgramError`` fallback handles them.\n    :class:`ProgramError`"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramError(
            "non-numeric result value / нечисловое значение результата"
        )
    if isinstance(value, int):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text or "0"
