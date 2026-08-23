import pytest

from app.core.db import DataGateway
from app.core.principal import resolve_principal
from app.core.proposals import ProposalStore
from app.knowledge.retrieval import ClauseIndex
from app.knowledge.rules import get_rules


@pytest.fixture(scope="session")
def rules():
    return get_rules()


@pytest.fixture(scope="session")
def index():
    return ClauseIndex()


@pytest.fixture()
def gateway(tmp_path):
    # A fresh database per test: the action tools write, and tests must not
    # observe each other's escalations.
    return DataGateway(db_path=tmp_path / "test.db", rebuild=True)


@pytest.fixture()
def runtime(gateway, rules, index):
    from app.agent.tools import ToolRuntime
    return ToolRuntime(gateway, rules, index, ProposalStore())


@pytest.fixture()
def now(gateway):
    return gateway.clock.now()


@pytest.fixture()
def northstar():
    return resolve_principal("cust-northstar")


@pytest.fixture()
def lumenworks():
    return resolve_principal("cust-lumenworks")


@pytest.fixture()
def agent_user():
    return resolve_principal("staff-rohit")


@pytest.fixture()
def manager():
    return resolve_principal("staff-priya")
