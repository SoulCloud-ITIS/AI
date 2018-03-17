from pathlib import Path

class FileConstants(object):

    base_path = Path(__file__).parents[1]

    @property
    def MODEL_ADVENTURE(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('models/adventure_model.doc2vec'))

    @property
    def MODEL_ART(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('models/art_model.doc2vec'))

    @property
    def MODEL_DETECTIVE(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('models/detective_model.doc2vec'))

    @property
    def MODEL_FANTASTIC(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('models/fantastic_model.doc2vec'))

    @property
    def MODEL_FANTASY(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('models/fantasy_model.doc2vec'))

    @property
    def MODEL_LOVE(self) -> str:
        return str( Path(str(FileConstants.base_path) ).joinpath('models/love_model.doc2vec'))

    @property
    def ADVENTURE_BOOK_TRAIN(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('books/train_books/adventure'))

    @property
    def ART_BOOK_TRAIN(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('books/train_books/art'))

    @property
    def DETECTIVE_BOOK_TRAIN(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('books/train_books/detective'))

    @property
    def FANTASTIC_BOOK_TRAIN(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('books/train_books/fantastic'))

    @property
    def FANTASY_BOOK_TRAIN(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('books/train_books/fantasy'))

    @property
    def LOVE_BOOK_TRAIN(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('books/train_books/love'))
