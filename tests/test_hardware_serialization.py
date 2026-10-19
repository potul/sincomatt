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

    def test_receptive_field_lookback_and_finetuning_loss(self):
        from audiotrain.data.dataset import AudioFrameDataset
        from audiotrain.losses.spectral import SpectralESRLoss

        am4 = AM4Model()
        am3 = AM3Model()
        ir1024 = IRModel(filter_size=1024)
        self.assertEqual(am4.receptive_field_pad, 1022)
        self.assertEqual(am3.receptive_field_pad, 573)
        self.assertEqual(ir1024.receptive_field_pad, 1023)

        # Verify AudioFrameDataset provides (512 + receptive_field_pad) dry samples
        dry = np.arange(4096, dtype=np.float32)
        wet = np.arange(4096, dtype=np.float32) * 2.0
        ds = AudioFrameDataset(dry, wet, frame_size=512, receptive_field_pad=am4.receptive_field_pad)
        self.assertEqual(len(ds), 8)

        # Frame 0 should have 1022 leading zeros + 512 samples [0..511]
        x0, y0 = ds[0]
        self.assertEqual(x0.shape, (1, 1534))
        self.assertEqual(y0.shape, (1, 512))
        self.assertTrue(torch.all(x0[0, :1022] == 0.0))
        self.assertTrue(torch.allclose(x0[0, 1022:], torch.arange(512, dtype=torch.float32)))

        # Frame 2 (starts at 1024) should have zero artificial padding, purely continuous [2..1535]
        x2, y2 = ds[2]
        self.assertEqual(x2.shape, (1, 1534))
        self.assertTrue(torch.allclose(x2[0], torch.arange(2, 1536, dtype=torch.float32)))

        # Verify AM4 valid convolution on 1534-sample input produces 512-sample output
        with torch.no_grad():
            out_am4 = am4(x2.unsqueeze(0), target_len=512)
            self.assertEqual(out_am4.shape, (1, 1, 512))

        # Verify SpectralESRLoss runs 2nd-order pre-emphasis + irfft(abs(rfft))
        spec_loss = SpectralESRLoss()
        pred = torch.randn(2, 1, 512)
        target = pred + 0.05 * torch.randn(2, 1, 512)
        loss_val = spec_loss(pred, target)
        self.assertTrue(torch.isfinite(loss_val))
        self.assertGreater(loss_val.item(), 0.0)

    def test_exact_microbatch_gradient_accumulation(self):
        import copy
        from audiotrain.losses.curriculum import CurriculumLoss

        for epoch in (1, 900):  # Test both BasicTuning (epoch=1) and FineTuning (epoch=900)
            torch.manual_seed(123)
            m_full = AM4Model()
            m_micro = copy.deepcopy(m_full)
            criterion = CurriculumLoss(fine_tuning_epoch=801)
            criterion.set_epoch(epoch)

            x = torch.randn(16, 1, 1534)
            y = torch.randn(16, 1, 512)

            # Full batch forward + backward
            pred_full = m_full(x, target_len=512)
            loss_full = criterion(pred_full, y)
            loss_full.backward()

            # Micro-batched (chunk=4) forward + backward
            with torch.no_grad():
                y_trans = criterion.transform(y)
                denom = torch.sum(y_trans ** 2) + criterion.eps
            loss_micro_val = 0.0
            for start in range(0, 16, 4):
                x_c = x[start : start + 4]
                y_c_trans = y_trans[start : start + 4]
                pred_c = m_micro(x_c, target_len=512)
                pred_c_trans = criterion.transform(pred_c)
                c_loss = torch.sum((y_c_trans - pred_c_trans) ** 2) / denom
                c_loss.backward()
                loss_micro_val += c_loss.item()

            self.assertAlmostEqual(loss_full.item(), loss_micro_val, places=6)
            for p1, p2 in zip(m_full.parameters(), m_micro.parameters()):
                self.assertTrue(torch.allclose(p1.grad, p2.grad, atol=1e-6))

    def test_all_modes_end_to_end(self):
        import tempfile
        import soundfile as sf
        from audiotrain.config import AppConfig, ModelType
        from audiotrain.trainer import AudioTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            sr = 44100
            # Create a short 0.5s stereo WAV for fast end-to-end testing of all modes
            t = np.linspace(0, 0.5, sr // 2, dtype=np.float32)
            dry = 0.5 * np.sin(2 * np.pi * 220.0 * t)
            wet = np.tanh(2.0 * dry)
            wav_file = tmp_p / "synth.wav"
            sf.write(str(wav_file), np.stack([dry, wet], axis=1), sr)

            expected_sizes = {
                ModelType.AM3: 60 + 1536 * 4,     # 6,204 bytes
                ModelType.AM4: 60 + 1952 * 4,     # 7,868 bytes
                ModelType.IR1024: 60 + 1024 * 4,  # 4,156 bytes
                ModelType.IR2048: 60 + 2048 * 4,  # 8,252 bytes
            }

            for mtype, exp_bytes in expected_sizes.items():
                cfg = AppConfig()
                cfg.samples_path = str(wav_file)
                cfg.output_dir = str(tmp_p / mtype.name)
                cfg.model.model_type = mtype
                cfg.model.model_name = f"test_{mtype.name.lower()}"
                cfg.apply_model_defaults()
                cfg.training.epochs = 2
                cfg.training.batch_size = 16
                cfg.training.device = "cpu"
                cfg.render_audio = True

                trainer = AudioTrainer(cfg)
                trained_model = trainer.train()
                self.assertIsNotNone(trained_model)

                out_files = list(Path(cfg.output_dir).glob("*Data"))
                self.assertEqual(len(out_files), 1, f"Expected 1 binary file for {mtype.name}")
                self.assertEqual(out_files[0].stat().st_size, exp_bytes, f"Size mismatch for {mtype.name}")

            # Test NAM_TO_AM3 and NAM_TO_AM4 if Fender.nam exists
            nam_file = Path("Fender/Fender.nam")
            if nam_file.is_file():
                for mtype, exp_bytes in ((ModelType.NAM_TO_AM3, 6204), (ModelType.NAM_TO_AM4, 7868)):
                    cfg = AppConfig()
                    cfg.samples_path = str(nam_file)
                    cfg.nam_path = str(nam_file)
                    cfg.dry_path = str(wav_file)
                    cfg.output_dir = str(tmp_p / mtype.name)
                    cfg.model.model_type = mtype
                    cfg.model.model_name = f"test_{mtype.name.lower()}"
                    cfg.apply_model_defaults()
                    cfg.training.epochs = 2
                    cfg.training.batch_size = 16
                    cfg.training.device = "cpu"
                    cfg.render_audio = False

                    trainer = AudioTrainer(cfg)
                    trained_model = trainer.train()
                    out_files = list(Path(cfg.output_dir).glob("*Data"))
                    self.assertEqual(len(out_files), 1)
                    self.assertEqual(out_files[0].stat().st_size, exp_bytes)
    def test_nam_vs_legacy_reamp_detection_and_segmentation(self):
        import soundfile as sf
        from audiotrain.data.audio import LoadedAudio, resample_signal
        from audiotrain.data.dataset import create_train_val_datasets, detect_reamp_profile

        # 1. Verify bundled default_dry.wav (NAM v3_0_0.wav) is detected at both 48 kHz and 44.1 kHz
        nam_asset = Path("audiotrain/assets/default_dry.wav")
        self.assertTrue(nam_asset.is_file())
        dry_48k, sr_48k = sf.read(str(nam_asset), dtype="float32")
        self.assertEqual(detect_reamp_profile(dry_48k, sr_48k), "NAM_V3")

        dry_44k = resample_signal(dry_48k, sr_48k, 44100)
        self.assertEqual(detect_reamp_profile(dry_44k, 44100), "NAM_V3")

        loaded_nam = LoadedAudio(
            dry=dry_44k,
            target=dry_44k,
            sample_rate=44100,
            dry_loudness=1.0,
            target_loudness=1.0,
            total_samples=len(dry_44k),
        )
        train_ds, val_ds = create_train_val_datasets(loaded_nam, frame_size=512)
        self.assertEqual(getattr(train_ds, "reamp_profile", None), "NAM_V3")
        self.assertEqual(train_ds.start_frame, (10 * 44100) // 512)
        self.assertEqual(len(train_ds), (181 * 44100) // 512 - (10 * 44100) // 512)
        self.assertEqual(len(val_ds), (190 * 44100) // 512 - (181 * 44100) // 512)

        # 2. Verify legacy ~59s reamp falls back to LEGACY_60_40 (60% train / 40% val)
        legacy_dry = np.sin(np.linspace(0, 1000, 2599989, dtype=np.float32))
        self.assertEqual(detect_reamp_profile(legacy_dry, 44100), "LEGACY_60_40")
        loaded_leg = LoadedAudio(
            dry=legacy_dry,
            target=legacy_dry,
            sample_rate=44100,
            dry_loudness=1.0,
            target_loudness=1.0,
            total_samples=len(legacy_dry),
        )
        tr_leg, va_leg = create_train_val_datasets(loaded_leg, frame_size=512)
        self.assertEqual(getattr(tr_leg, "reamp_profile", None), "LEGACY_60_40")
        total_f = len(legacy_dry) // 512
        self.assertEqual(len(tr_leg), int(total_f * 0.6))
        self.assertEqual(len(va_leg), total_f - int(total_f * 0.6))


if __name__ == "__main__":
    unittest.main()
