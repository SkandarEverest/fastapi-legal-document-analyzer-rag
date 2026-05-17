from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from app.core.settings import settings

engine = create_engine(settings.SQLITE_URL, echo=False, connect_args={"check_same_thread": False})


def init_db() -> None:
    from app.models import chunk, document  # noqa: F401  register models

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
