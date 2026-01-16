"""
Unit and Integration Tests for Inference Throughput Benchmarking

This module tests the benchmarking infrastructure for measuring and optimizing
PtychoPINN inference throughput.
"""

import unittest
import sys
import tempfile
import time
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.benchmark_inference_throughput import (
    BenchmarkConfig, TimingProfiler, MemoryProfiler, 
    ThroughputAnalyzer, InferenceBenchmark
)


class TestBenchmarkConfig(unittest.TestCase):
    """Test configuration dataclass for benchmarking."""
    
    def test_config_initialization(self):
        """Test BenchmarkConfig initialization with default values."""
        config = BenchmarkConfig(
            model_path=Path("/path/to/model"),
            test_data_path=Path("/path/to/data.npz")
        )
        
        self.assertEqual(config.model_path, Path("/path/to/model"))
        self.assertEqual(config.test_data_path, Path("/path/to/data.npz"))
        self.assertEqual(config.batch_sizes, [1, 4, 8, 16, 32, 64, 128])
        self.assertEqual(config.n_images, 1000)
        self.assertEqual(config.num_runs, 5)
        self.assertEqual(config.warmup_runs, 3)
        self.assertTrue(config.profile_memory)
        self.assertTrue(config.plot_results)
    
    def test_config_custom_values(self):
        """Test BenchmarkConfig with custom values."""
        config = BenchmarkConfig(
            model_path=Path("/custom/model"),
            test_data_path=Path("/custom/data.npz"),
            batch_sizes=[2, 4, 8],
            n_images=500,
            num_runs=3,
            warmup_runs=1,
            profile_memory=False,
            adaptive_sizing=True
        )
        
        self.assertEqual(config.batch_sizes, [2, 4, 8])
        self.assertEqual(config.n_images, 500)
        self.assertEqual(config.num_runs, 3)
        self.assertEqual(config.warmup_runs, 1)
        self.assertFalse(config.profile_memory)
        self.assertTrue(config.adaptive_sizing)


class TestTimingProfiler(unittest.TestCase):
    """Test timing profiler functionality."""
    
    def setUp(self):
        self.profiler = TimingProfiler()
    
    def test_measure_function(self):
        """Test measuring execution time of a function."""
        def test_func(x):
            time.sleep(0.01)  # Sleep for 10ms
            return x * 2
        
        result, elapsed = self.profiler.measure("test_op", test_func, 5)
        
        self.assertEqual(result, 10)
        self.assertGreater(elapsed, 0.009)  # Should be at least 9ms
        self.assertLess(elapsed, 0.02)  # Should be less than 20ms
    
    def test_get_statistics(self):
        """Test calculating timing statistics."""
        # Add multiple timing measurements
        for i in range(5):
            self.profiler.timings["test_op"].append(0.01 * (i + 1))
        
        stats = self.profiler.get_statistics("test_op")
        
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('median', stats)
        
        self.assertAlmostEqual(stats['mean'], 0.03, places=3)
        self.assertAlmostEqual(stats['min'], 0.01, places=3)
        self.assertAlmostEqual(stats['max'], 0.05, places=3)
        self.assertAlmostEqual(stats['median'], 0.03, places=3)
    
    def test_clear_timings(self):
        """Test clearing timing data."""
        self.profiler.timings["test_op"] = [0.01, 0.02]
        self.profiler.clear()
        
        self.assertEqual(len(self.profiler.timings), 0)


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiling functionality."""
    
    def setUp(self):
        self.profiler = MemoryProfiler()
    
    def test_profile_cpu_memory(self):
        """Test CPU memory profiling."""
        memory_mb = self.profiler.profile_cpu_memory()
        
        # Should return a positive value (current process memory)
        self.assertGreater(memory_mb, 0)
        self.assertLess(memory_mb, 10000)  # Reasonable upper bound
    
    def test_snapshot(self):
        """Test taking memory snapshots."""
        snapshot = self.profiler.snapshot("test_label")
        
        self.assertIn('timestamp', snapshot)
        self.assertIn('label', snapshot)
        self.assertIn('cpu_mb', snapshot)
        self.assertEqual(snapshot['label'], "test_label")
        self.assertGreater(snapshot['cpu_mb'], 0)
        
        # Check snapshot is stored
        self.assertEqual(len(self.profiler.memory_snapshots), 1)
        self.assertEqual(self.profiler.memory_snapshots[0], snapshot)


class TestThroughputAnalyzer(unittest.TestCase):
    """Test throughput analysis functionality."""
    
    def setUp(self):
        self.analyzer = ThroughputAnalyzer()
    
    def test_calculate_throughput(self):
        """Test throughput calculation."""
        throughput = self.analyzer.calculate_throughput(n_images=100, elapsed_time=2.0)
        self.assertEqual(throughput, 50.0)  # 100 images / 2 seconds = 50 img/s
        
        # Test edge case
        throughput = self.analyzer.calculate_throughput(n_images=100, elapsed_time=0)
        self.assertEqual(throughput, 0.0)
    
    def test_calculate_latency(self):
        """Test latency calculation."""
        latency = self.analyzer.calculate_latency(elapsed_time=1.0, batch_size=10)
        self.assertEqual(latency, 100.0)  # 1s / 10 images * 1000 = 100ms per image
        
        # Test edge case
        latency = self.analyzer.calculate_latency(elapsed_time=1.0, batch_size=0)
        self.assertEqual(latency, 0.0)
    
    def test_calculate_efficiency(self):
        """Test efficiency calculation."""
        efficiency = self.analyzer.calculate_efficiency(throughput=100.0, memory_usage=50.0)
        self.assertEqual(efficiency, 2.0)  # 100 img/s / 50 MB = 2 img/s/MB
        
        # Test edge case
        efficiency = self.analyzer.calculate_efficiency(throughput=100.0, memory_usage=0)
        self.assertEqual(efficiency, 0.0)
    
    def test_find_optimal_batch_size(self):
        """Test finding optimal batch size from results."""
        results = {
            8: {'status': 'success', 'throughput': 100},
            16: {'status': 'success', 'throughput': 150},
            32: {'status': 'success', 'throughput': 120},
            64: {'status': 'OOM', 'throughput': 0}
        }
        
        optimal_batch, optimal_result = self.analyzer.find_optimal_batch_size(results)
        
        self.assertEqual(optimal_batch, 16)
        self.assertEqual(optimal_result['throughput'], 150)
    
    def test_find_optimal_batch_size_no_success(self):
        """Test finding optimal batch size with no successful runs."""
        results = {
            32: {'status': 'OOM'},
            64: {'status': 'error'}
        }
        
        optimal_batch, optimal_result = self.analyzer.find_optimal_batch_size(results)
        
        self.assertEqual(optimal_batch, 0)
        self.assertEqual(optimal_result, {})



class TestInferenceBenchmarkIntegration(unittest.TestCase):
    """Integration tests for the complete benchmark workflow."""
    
    @patch('scripts.benchmark_inference_throughput.loader.load')
    @patch('scripts.benchmark_inference_throughput.load_inference_bundle')
    @patch('scripts.benchmark_inference_throughput.RawData')
    @patch('numpy.load')
    def test_load_model_and_data(self, mock_np_load, mock_raw_data_class, mock_load_bundle, mock_loader_load):
        """Test loading model and data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mocks
            mock_model = Mock()
            mock_config = {
                'obj_w': 232,
                'obj_h': 232, 
                'intensity_scale': 1.0
            }
            mock_load_bundle.return_value = (mock_model, mock_config)
            
            # Setup mock numpy.load to return data structure expected by RawData
            mock_npz_data = {
                'diffraction': np.random.randn(100, 64, 64),
                'xcoords': np.random.randn(100),
                'ycoords': np.random.randn(100),
                'probeGuess': np.random.randn(64, 64),
                'objectGuess': np.random.randn(232, 232)
            }
            mock_np_load.return_value = mock_npz_data
            
            # Setup mock raw data instance
            mock_raw_data = Mock()
            mock_raw_data.diffraction = np.random.randn(100, 64, 64)
            mock_raw_data.xcoords = np.random.randn(100)
            mock_raw_data.ycoords = np.random.randn(100)
            mock_raw_data.probeGuess = np.random.randn(64, 64)
            mock_raw_data.objectGuess = np.random.randn(232, 232)
            mock_raw_data.generate_grouped_data = Mock(return_value={
                'diffraction': np.random.randn(50, 64, 64, 1),
                'coords_offsets': np.random.randn(50, 1, 2, 1),
                'coords_relative': np.random.randn(50, 1, 2, 1),
                'nn_indices': np.random.randint(0, 100, (50, 1)),
                'X_full': np.random.randn(50, 64, 64, 1)
            })
            
            # Configure RawData class constructor to return mock instance
            mock_raw_data_class.return_value = mock_raw_data
            
            # Setup mock test data container
            mock_test_data = Mock()
            mock_test_data.X = np.random.randn(50, 64, 64)
            mock_test_data.local_offsets = np.random.randn(50, 2)
            mock_test_data.global_offsets = np.random.randn(50, 2)
            mock_loader_load.return_value = mock_test_data
            
            # Create config
            config = BenchmarkConfig(
                model_path=Path(tmpdir) / "model",
                test_data_path=Path(tmpdir) / "data.npz",
                n_images=50,
                output_dir=Path(tmpdir) / "output"
            )
            
            # Create benchmark instance
            benchmark = InferenceBenchmark(config)
            
            # Load model and data
            model, test_data = benchmark.load_model_and_data()
            
            # Verify
            self.assertEqual(model, mock_model)
            self.assertEqual(len(test_data.X), 50)  # Should limit to n_images
            mock_raw_data.generate_grouped_data.assert_called_once()
    
    def test_warmup_inference(self):
        """Test warmup inference runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock model
            mock_model = Mock()
            mock_model.predict = Mock(return_value=np.random.randn(32, 64, 64))
            
            # Setup mock test data
            mock_test_data = Mock()
            mock_test_data.X = np.random.randn(100, 64, 64)
            mock_test_data.local_offsets = np.random.randn(100, 2)
            
            # Create config
            config = BenchmarkConfig(
                model_path=Path(tmpdir) / "model",
                test_data_path=Path(tmpdir) / "data.npz",
                output_dir=Path(tmpdir) / "output"
            )
            
            # Create benchmark instance
            benchmark = InferenceBenchmark(config)
            
            # Run warmup
            benchmark.warmup_inference(mock_model, mock_test_data, warmup_runs=3)
            
            # Verify predict was called 3 times
            self.assertEqual(mock_model.predict.call_count, 3)
    
    def test_generate_summary(self):
        """Test summary generation from results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                model_path=Path(tmpdir) / "model",
                test_data_path=Path(tmpdir) / "data.npz",
                output_dir=Path(tmpdir) / "output"
            )
            
            benchmark = InferenceBenchmark(config)
            
            # Create mock results
            benchmark.results = {
                8: {
                    'status': 'success',
                    'statistics': {
                        'throughput': {'mean': 100, 'std': 5},
                        'latency': {'mean': 10, 'std': 1}
                    },
                    'memory': {'cpu_mb': 500}
                },
                16: {
                    'status': 'success',
                    'statistics': {
                        'throughput': {'mean': 150, 'std': 8},
                        'latency': {'mean': 6.7, 'std': 0.5}
                    },
                    'memory': {'cpu_mb': 800}
                },
                32: {
                    'status': 'OOM'
                }
            }
            
            # Generate summary
            summary = benchmark.generate_summary(optimal_batch_size=16)
            
            # Verify summary contents
            self.assertEqual(summary['optimal_batch_size'], 16)
            self.assertEqual(summary['max_throughput'], 150)
            self.assertAlmostEqual(summary['speedup_vs_batch_1'], 1.5, places=1)
            self.assertIn('timestamp', summary)
            self.assertIn('config', summary)
            self.assertIn('results_by_batch_size', summary)
    
    def test_export_results(self):
        """Test exporting results to JSON and CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            config = BenchmarkConfig(
                model_path=Path(tmpdir) / "model",
                test_data_path=Path(tmpdir) / "data.npz",
                output_dir=output_dir,
                export_formats=['json', 'csv']
            )
            
            benchmark = InferenceBenchmark(config)
            
            # Create mock summary
            summary = {
                'optimal_batch_size': 16,
                'max_throughput': 150,
                'results_by_batch_size': {
                    8: {
                        'status': 'success',
                        'statistics': {
                            'throughput': {'mean': 100, 'std': 5},
                            'latency': {'mean': 10, 'std': 1}
                        },
                        'memory': {'cpu_mb': 500}
                    },
                    16: {
                        'status': 'success',
                        'statistics': {
                            'throughput': {'mean': 150, 'std': 8},
                            'latency': {'mean': 6.7, 'std': 0.5}
                        },
                        'memory': {'cpu_mb': 800}
                    }
                }
            }
            
            # Export results
            benchmark.export_results(summary)
            
            # Check JSON export
            json_path = output_dir / 'benchmark_results.json'
            self.assertTrue(json_path.exists())
            
            with open(json_path, 'r') as f:
                loaded_json = json.load(f)
                self.assertEqual(loaded_json['optimal_batch_size'], 16)
            
            # Check CSV export
            csv_path = output_dir / 'benchmark_results.csv'
            self.assertTrue(csv_path.exists())


class TestPerformanceRegression(unittest.TestCase):
    """Test for performance regression in benchmarking."""
    
    def test_timing_profiler_performance(self):
        """Test that timing profiler has minimal overhead."""
        profiler = TimingProfiler()
        
        def fast_func():
            return sum(range(1000))
        
        # Measure overhead
        start = time.perf_counter()
        for _ in range(1000):
            result, _ = profiler.measure("test", fast_func)
        elapsed_with_profiler = time.perf_counter() - start
        
        # Measure without profiler
        start = time.perf_counter()
        for _ in range(1000):
            result = fast_func()
        elapsed_without = time.perf_counter() - start
        
        # Overhead should be less than 20%
        overhead_ratio = elapsed_with_profiler / elapsed_without
        self.assertLess(overhead_ratio, 1.2, 
                       f"Profiler overhead too high: {overhead_ratio:.2f}x")


if __name__ == '__main__':
    unittest.main()