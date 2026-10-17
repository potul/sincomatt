import unittest
from pathlib import Path
import numpy as np
import torch

from audiotrain.models.am4 import AM4Model
from audiotrain.models.am3 import AM3Model
from audiotrain.models.ir import IRModel
from audiotrain.serialization.binary_format import (
    pack_am_data,
    unpack_am_data,
    build_header,
    parse_header,
    HEADER_SIZE,
    SIGNATURE_DEFAULT,
    CONTROL_WORD_DEFAULT,
)


class TestHardwareBinarySerialization(unittest.TestCase):
    def test_header_structure(self):
        hdr = build_header("AM4", "TestModel")
        self.assertEqual(len(hdr), HEADER_SIZE)
        self.assertEqual(hdr[:4], b"AM4\x00")
        self.assertEqual(hdr[4:18], b"TestModel\x00\x00\x00\x00\x00")
        self.assertEqual(hdr[18:20], b"\x64\x64")
        self.assertEqual(hdr[20:22], b"\x3c\x1e")
        self.assertEqual(hdr[22:24], b"\x00\x00")

        parsed = parse_header(hdr)
        self.assertEqual(parsed["magic"], "AM4")
        self.assertEqual(parsed["name"], "TestModel")
        self.assertEqual(parsed["version_sig"], SIGNATURE_DEFAULT)
        self.assertEqual(parsed["control_word"], CONTROL_WORD_DEFAULT)

    def test_am4_fender_exact_match(self):
        fender_file = Path("Fender/Fender.am4Data")
        if not fender_file.is_file():
            self.skipTest("Fender.am4Data not found")

        with open(fender_file, "rb") as f:
            orig = f.read()

        model = AM4Model()
        header_info, _ = unpack_am_data(orig, model=model)
        repacked = pack_am_data(
            model,
            model_name=header_info["name"],
            eq_settings={
                "version_sig": header_info["version_sig"],
                "control_word": header_info["control_word"],
                "reserved": header_info["reserved"],
                "bass_freq": header_info["eq_bass"][0],
                "bass_min": header_info["eq_bass"][1],
                "bass_max": header_info["eq_bass"][2],
                "mid_freq": header_info["eq_mid"][0],
                "mid_min": header_info["eq_mid"][1],
                "mid_max": header_info["eq_mid"][2],
                "treble_freq": header_info["eq_treble"][0],
                "treble_min": header_info["eq_treble"][1],
                "treble_max": header_info["eq_treble"][2],
            },
        )
        self.assertEqual(len(repacked), len(orig))
        self.assertEqual(orig, repacked)

    def test_am3_kpablue_exact_match(self):
        kpa_file = Path("SincoAnn/kpa-blue/kpa-blue.am3Data")
        if not kpa_file.is_file():
            self.skipTest("kpa-blue.am3Data not found")

        with open(kpa_file, "rb") as f:
            orig = f.read()

        model = AM3Model()
        header_info, _ = unpack_am_data(orig, model=model)
        repacked = pack_am_data(
            model,
            model_name=header_info["name"],
            eq_settings={
                "version_sig": header_info["version_sig"],
                "control_word": header_info["control_word"],
                "reserved": header_info["reserved"],
                "bass_freq": header_info["eq_bass"][0],
                "bass_min": header_info["eq_bass"][1],
                "bass_max": header_info["eq_bass"][2],
                "mid_freq": header_info["eq_mid"][0],
                "mid_min": header_info["eq_mid"][1],
                "mid_max": header_info["eq_mid"][2],
                "treble_freq": header_info["eq_treble"][0],
                "treble_min": header_info["eq_treble"][1],
                "treble_max": header_info["eq_treble"][2],
            },
        )
        self.assertEqual(len(repacked), len(orig))
        self.assertEqual(orig, repacked)

    def test_causal_padding_and_impulse_init(self):
        am4 = AM4Model()
        self.assertEqual(am4.fir1.weight[0, 0, -1].item(), 1.0)
        self.assertEqual(am4.fir2.weight[0, 0, -1].item(), 1.0)

        am3 = AM3Model()
        self.assertEqual(am3.fir1.weight[0, 0, -1].item(), 1.0)
        self.assertEqual(am3.fir2.weight[0, 0, -1].item(), 1.0)

        ir = IRModel(filter_size=1024)
        self.assertEqual(ir.fir.weight[0, 0, -1].item(), 1.0)

        # Verify causal IR pass-through has zero phase shift (y[t] == x[t])
        x = torch.randn(2, 1, 512)
        y_ir = ir(x, target_len=512)
        self.assertTrue(torch.allclose(x, y_ir, atol=1e-6))

        # Verify causality on AM4 and AM3: changing future samples x[:, :, t+1:] must not affect y[:, :, :t+1]
        x1 = torch.randn(1, 1, 512)
        x2 = x1.clone()
        x2[:, :, 256:] += 10.0

        with torch.no_grad():
            y1_am4 = am4(x1, target_len=512)
            y2_am4 = am4(x2, target_len=512)
            self.assertTrue(torch.allclose(y1_am4[:, :, :256], y2_am4[:, :, :256], atol=1e-6))

            y1_am3 = am3(x1, target_len=512)
            y2_am3 = am3(x2, target_len=512)
            self.assertTrue(torch.allclose(y1_am3[:, :, :256], y2_am3[:, :, :256], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

