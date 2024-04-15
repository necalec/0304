#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;

const int width = 1920;
const int height = 1080;
const double xmin = -2.0;
const double xmax = 1.0;
const double ymin = -1.5;
const double ymax = 1.5;
const int max_iter = 1000;

int mandelbrot(const double cr, const double ci, const int max_iter) {
    double zr = 0.0, zi = 0.0;
    int i = 0;
    while (i < max_iter && zr * zr + zi * zi < 4.0) {
        double temp = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = temp;
        ++i;
    }
    return i;
}

int main() {
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image(height, width, CV_8UC3, Scalar(0, 0, 0));

    int rows_per_proc = height / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? height : (rank + 1) * rows_per_proc;

    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < width; ++x) {
            double cr = xmin + (xmax - xmin) * x / (width - 1);
            double ci = ymin + (ymax - ymin) * y / (height - 1);
            int iter = mandelbrot(cr, ci, max_iter);
            double hue = iter == max_iter ? 0 : 4.0 * log(iter + 1.0) / log(max_iter + 1.0);
            uchar r = (uchar)(255 * hue);
            uchar g = (uchar)(255 * hue);
            uchar b = (uchar)(255 * hue);
            image.at<Vec3b>(y, x) = Vec3b(b, g, r);
        }
    }

    Mat result_image(height, width, CV_8UC3);
    MPI_Gather(image.data, width * rows_per_proc * 3, MPI_UNSIGNED_CHAR,
        result_image.data, width * rows_per_proc * 3, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        namedWindow("1", WINDOW_NORMAL);
        imshow("1", result_image);
        waitKey(0);
    }

    MPI_Finalize();
    return 0;
}
