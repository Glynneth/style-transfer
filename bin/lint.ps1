"`nisort:"
isort src tests

"`nblack:"
black -t py310 -l 79 src tests

"`nflake8:"
flake8 src tests --max-line-length=99

"`nmypy src:"
mypy src --ignore-missing-imports --disallow-untyped-defs

"`nmypy tests:"
mypy tests --ignore-missing-imports