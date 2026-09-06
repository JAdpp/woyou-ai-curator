"""Offline tests for the temporary CONNECT relay's strict destination boundary."""

import unittest

from aliyun_connect_relay import ALLOWED_AUTHORITY, MAX_HEADER_BYTES, request_allowed


class ConnectAllowlistTests(unittest.TestCase):
    def test_exact_authority_only(self):
        self.assertTrue(request_allowed(f"CONNECT {ALLOWED_AUTHORITY} HTTP/1.1\r\nHost: {ALLOWED_AUTHORITY}\r\n\r\n".encode()))
        self.assertTrue(request_allowed(f"CONNECT {ALLOWED_AUTHORITY} HTTP/1.0\r\n\r\n".encode()))

    def test_reject_other_targets_and_methods(self):
        for target in ("dashscope.aliyuncs.com:443", "api.deepseek.com:443", "127.0.0.1:443",
                       ALLOWED_AUTHORITY.replace(":443", ":80"), "user@" + ALLOWED_AUTHORITY,
                       ALLOWED_AUTHORITY.replace(":443", ".:443"), ALLOWED_AUTHORITY.upper()):
            with self.subTest(target=target):
                self.assertFalse(request_allowed(f"CONNECT {target} HTTP/1.1\r\n\r\n".encode()))
        self.assertFalse(request_allowed(f"GET https://{ALLOWED_AUTHORITY}/ HTTP/1.1\r\n\r\n".encode()))

    def test_reject_header_ambiguity_or_payload(self):
        for header in ("Host: another.example:443", "Content-Length: 1", "Transfer-Encoding: chunked",
                       " Host: ignored.example", "malformed"):
            with self.subTest(header=header):
                self.assertFalse(request_allowed(f"CONNECT {ALLOWED_AUTHORITY} HTTP/1.1\r\n{header}\r\n\r\n".encode()))

    def test_reject_incomplete_oversized_or_non_ascii(self):
        self.assertFalse(request_allowed(f"CONNECT {ALLOWED_AUTHORITY} HTTP/1.1\r\n".encode()))
        self.assertFalse(request_allowed(b"x" * MAX_HEADER_BYTES + b"\r\n\r\n"))
        self.assertFalse(request_allowed(b"\xff\r\n\r\n"))


if __name__ == "__main__":
    unittest.main()
