"`nisort:"
isort src

"`nflake8:"
flake8 src --max-line-length=99

"`nblack:"
black -t py310 -l 79 src

"`nmypy:"
mypy src --ignore-missing-imports --disallow-untyped-defs