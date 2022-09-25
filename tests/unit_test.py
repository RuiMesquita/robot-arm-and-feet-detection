import unittest
import os
import src.utils.functions as f


def add_fish_to_aquarium(fish_list):
    if len(fish_list) > 10:
        raise ValueError("A maximum of 10 fish can be added to the aquarium")
    return {"tank_a": fish_list}


def calculate_real_coordinates(x, y, w_real, h_real, w_pixel, h_pixel):
    if h_pixel != 0 and w_pixel != 0:
        x_real = round(x * (w_real/w_pixel), 3)
        y_real = round(y * (h_real/h_pixel), 3)
    else:
        x_real = 0
        y_real = 0

    return x_real, y_real


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def write_training_report(epoch_status):
    file = open("training_report.txt", "a")
    file.write(epoch_status)
    file.write("\n")
    file.close()


def write_metrics_report(metrics):
    file = open("metrics.txt", "a")
    file.write(metrics)
    file.close()


class TestCalculateRealCoordinates(unittest.TestCase):
    def test_calculate_real_coordinates_success(self):
        actual = calculate_real_coordinates(1, 1, 2, 2, 2, 2)
        expected = (1.000, 1.000)
        self.assertEqual(actual, expected)

    def test_calculate_real_coordinates_failure(self):
        actual = calculate_real_coordinates(1, 1, 0, 0, 0, 0)
        expected = (0, 0)
        self.assertEqual(actual, expected)


class TestMakeDir(unittest.TestCase):
    def test_make_dir_success(self):
        directory_path = os.path.join("tests", "mock_dir")
        make_dir(directory_path)
        self.assertTrue(os.path.exists(directory_path))

    @classmethod
    def tearDownClass(self):
        directory_path = os.path.join("tests", "mock_dir")
        os.rmdir(directory_path)


class TestEpochTime(unittest.TestCase):
    def test_epoch_time_success(self):
        start_time, end_time = 1664110800, 1664114400
        actual = epoch_time(start_time, end_time)
        expected = 60, 0
        self.assertEqual(actual, expected)


class TestWriteTrainingReport(unittest.TestCase):
    def test_write_training_report_success(self):
        epoch_status = f'Epoch: 0 | Epoch Time: 0m 0s\n'
        epoch_status += f'\tTrain Loss: 0\n'
        epoch_status += f'\t Val. Loss: 0\n'
        write_training_report(epoch_status)
        self.assertTrue(os.path.exists("./training_report.txt"))
        self.assertNotEqual(open("./training_report.txt").read(), "")

    @classmethod
    def tearDownClass(self):
        os.remove("./training_report.txt")


class TestWriteMetricsReport(unittest.TestCase):
    def test_write_metrics_report_success(self):
        metrics = f'Jaccard: 0.60'
        write_metrics_report(metrics)
        self.assertTrue(os.path.exists("./metrics.txt"))
        self.assertNotEqual(open("./metrics.txt").read(), "")

    @classmethod
    def tearDownClass(self):
        os.remove("./metrics.txt")


class TestGenerateGraphReport(unittest.TestCase):
    def test_generate_graph_report_success(self):
        num_epochs = 10
        train_loss = [1] * 10
        valid_loss = [2] * 10
        f.generate_graph_report(num_epochs, train_loss, valid_loss)
        self.assertTrue(os.path.exists("training_graph.png"))

    @classmethod
    def tearDownClass(self):
        os.remove("./training_graph.png")
