from enum import Enum


class Test(str, Enum):
    upload = "upload"
    download = "download"
    bidir = "bidir"
    latency = "latency"
    file_download = "file-download"
    file_upload = "file-upload"
    browsing = "browsing"
