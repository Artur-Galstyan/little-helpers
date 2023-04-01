import nox


@nox.session
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "tests")


@nox.session
def lint(session):
    # use Ruff linting
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "ruff", "little_helpers", "tests")

