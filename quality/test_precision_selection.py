"""Check graph selection for old packages and co-located precision variants."""
import tempfile
import unittest
from pathlib import Path

from clients.ctc_transcribe import select_model
from clients.rnnt_transcribe import select_encoder


class GraphSelectionTest(unittest.TestCase):
    def test_ctc_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                select_model(root)
            floating = root / "model.fp32.onnx"
            floating.touch()
            self.assertEqual(select_model(root), floating)
            (root / "model.int4.onnx").touch()
            with self.assertRaisesRegex(ValueError, "--model"):
                select_model(root)
            default = root / "model.int8.onnx"
            default.touch()
            self.assertEqual(select_model(root), default)

    def test_ctc_sherpa_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp32 = root / "model.onnx"
            fp32.touch()
            self.assertEqual(select_model(root), fp32)
            (root / "model.fp16.onnx").touch()
            with self.assertRaisesRegex(ValueError, "--model"):
                select_model(root)
            int8 = root / "model.int8.onnx"
            int8.touch()
            self.assertEqual(select_model(root), int8)

    def test_rnnt_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "v4_rnnt"
            with self.assertRaises(FileNotFoundError):
                select_encoder(root, prefix, {})
            floating = root / "v4_rnnt_encoder.onnx"
            floating.touch()
            self.assertEqual(select_encoder(root, prefix, {}), floating)
            (root / "v4_rnnt_encoder.int4.onnx").touch()
            with self.assertRaisesRegex(ValueError, "--encoder"):
                select_encoder(root, prefix, {})
            default = root / "v4_rnnt_encoder.int8.onnx"
            default.touch()
            self.assertEqual(select_encoder(root, prefix, {}), default)
            manifest = {"encoder_filename": floating.name}
            self.assertEqual(select_encoder(root, prefix, manifest), floating)


if __name__ == "__main__":
    unittest.main()
