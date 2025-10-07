from application.parser.remote.base import BaseRemote
from langchain.document_loader import TelegramChatApiLoader

class TelegramChatApiRemote(BaseRemote):
    def _init_parser(self, *args, **load_kwargs):
        self.loader = TelegramChatApiLoader(**load_kwargs)
        return {}

    def parse_file(self, *args, **load_kwargs):

        return 
