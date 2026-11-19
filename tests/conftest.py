import pytest
from dishka import Provider, Scope
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.di.providers import create_container
from app.domain.ports import DocumentParser, Embedder, LLMClientFactory, Reranker, UrlResolver, VectorStore
from app.infra.parsing import LangChainParser
from tests.stubs import (
    DIM,
    FakeEmbedder,
    FakeLLM,
    FakeLLMFactory,
    FakeReranker,
    FakeUrlResolver,
    FakeWebLoader,
    InMemoryVectorStore,
)

UA_DOCS = {
    "phishing.txt": (
        "Фішинг — це вид кібератаки, коли зловмисник надсилає підроблені листи, щоб викрасти паролі. "
        "Щоб захиститися від фішингу, перевіряйте адресу відправника та не відкривайте підозрілі вкладення."
    ),
    "mfa.txt": (
        "Двофакторна автентифікація значно підвищує безпеку облікових записів. "
        "Навіть якщо пароль викрадено, зловмисник не зможе увійти без другого фактора."
    ),
    "backup.txt": (
        "Резервне копіювання даних захищає від програм-вимагачів. "
        "Зберігайте копії офлайн і регулярно перевіряйте їх відновлення."
    ),
}


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        vector_store="qdrant",  # overridden by InMemoryVectorStore below
        embedding_dim=DIM,
        llm_provider="vllm",
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def test_provider(fake_llm, fake_reranker) -> Provider:
    """Overrides the GPU / network dependencies with in-process fakes."""
    provider = Provider(scope=Scope.APP)
    provider.provide(lambda: InMemoryVectorStore(), provides=VectorStore, override=True)
    provider.provide(lambda: FakeEmbedder(), provides=Embedder, override=True)
    provider.provide(lambda: fake_reranker, provides=Reranker, override=True)
    provider.provide(lambda: FakeLLMFactory(fake_llm), provides=LLMClientFactory, override=True)
    provider.provide(lambda: FakeUrlResolver(), provides=UrlResolver, override=True)
    provider.provide(
        lambda: LangChainParser(200, 20, web_loader_factory=FakeWebLoader),
        provides=DocumentParser,
        override=True,
    )
    return provider


@pytest.fixture
async def container(settings, test_provider):
    container = create_container(settings, test_provider)
    yield container
    await container.close()


@pytest.fixture
async def client(settings, test_provider):
    from app.main import create_app

    app = create_app(settings, create_container(settings, test_provider))
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test/api/v1") as c,
    ):
        yield c
