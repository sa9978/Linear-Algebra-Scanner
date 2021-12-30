import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy.linalg

def get_input(file_name):
    img = Image.open(file_name)
    img = np.asarray(img)
    img = to_mtx(img)
    return img


def to_mtx(img):
    """
    This method just reverse x and y of an image matrix because of the different order of x and y in PIL and Matplotlib library
    """
    H, V, C = img.shape
    mtr = np.zeros((V, H, C), dtype='int')
    for i in range(img.shape[0]):
        mtr[:, i] = img[i]
    return mtr


def get_coef(a, b, n):
    res = []
    b = [b[0], b[1], 1]
    dim = 3
    for i in range(dim):
        curr = [0] * dim * 4
        curr[i] = a[0]
        curr[dim + i] = a[1]
        curr[2 * dim + i] = 1 if i != 2 else 0

        curr[3 * dim + n - 1] = -b[i]
        res.append(curr)

    return res


def getPerspectiveTransform(pts1, pts2):
    A = []
    plen = len(pts1)

    for i in range(plen):
        A += get_coef(pts1[i], pts2[i], i)

    B = [0, 0, -1] * plen
    C = np.linalg.solve(A, B)

    res = np.ones(9)
    res[:8] = C.flatten()[:8]

    return res.reshape(3, -1).T


def showWarpPerspective(dst):
    width, height, _ = dst.shape

    # This part is for denoting the result matrix . You can use this if at first you have filled matrix with zeros
    for i in range(width - 1, -1, -1):
        for j in range(height - 1, -1, -1):
            if dst[i][j][0] == 0 and dst[i][j][1] == 0 and dst[i][j][2] == 0:
                if i + 1 < width and j - 1 >= 0:
                    dst[i][j] = dst[i + 1][j - 1]

    showImage(dst, title='Warp Perspective')


def showImage(image, title, save_file=True):
    final_ans = to_mtx(image)
    final_ans = final_ans.astype(np.uint8)

    plt.title(title)
    plt.imshow(final_ans)

    if save_file:
        try:
            os.mkdir('out')
        except OSError:
            pass
        path = os.path.join('out', title + '.jpg')
        plt.savefig(path, bbox_inches='tight')

    plt.show()


def Filter(img, filter_matrix):
    m, n, l = img.shape
    res = np.zeros((m, n, l))

    for i in range(m):
        for j in range(n):
            reshaped = np.reshape(img[i, j, :], newshape=(3,))
            res[i, j, :] = filter_matrix.dot(reshaped)

    return res


def warpPerspective(img, transform_matrix, output_width, output_height):
    """
    TODO : find warp perspective of image_matrix and return it
    :return a (width x height) warped image
    """
    res = np.zeros((output_width, output_height, 3))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            dot_matrix = np.dot(transform_matrix, [x, y, 1])
            t1 = int(dot_matrix[0] / dot_matrix[2])
            t2 = int(dot_matrix[1] / dot_matrix[2])
            if t1 < output_width and t2 < output_height:
                res[t1, t2, :] = img[x, y, :]
    return res[:output_width, :output_height, :]


def grayScaledFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """
    matrix = np.array([[0.2999, 0.587, 0.114], [0.2999, 0.587, 0.114], [0.2999, 0.587, 0.114]])
    return Filter(img, matrix)


def crazyFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """

    matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]])
    return Filter(img, matrix)


def customFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """
    matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 1]])
    custom_photo = Filter(img, matrix)
    showImage(custom_photo, title="Custom Filter")
    inverse = numpy.linalg.inv(matrix)
    new_image = Filter(custom_photo, inverse)
    showImage(new_image, title="First Photo")


def scaleImg(img, scale_width, scale_height):
    """
    TODO : Complete this part based on the description in the manual!
    """
    width1 = img.shape[0] * scale_width
    height1 = img.shape[1] * scale_height
    res = np.zeros((width1, height1, 3))
    for x in range(width1):
        for y in range(height1):
            res[x, y, :] = img[int(x / scale_width), int(y / scale_height), :]
    return res

def cropImg(img, start_row, end_row, start_column, end_column):  # 50, 300, 50, 225
    """
    TODO : Complete this part based on the description in the manual!
    """
    res = np.zeros((end_column - start_column, end_row - start_row, 3))
    for x in range(end_column - start_column):
        for y in range(end_row - start_row):
            res[x, y, :] = img[x + start_column, y + start_row, :]
    return res


if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    # You can change width and height if you want
    width, height = 300, 400

    showImage(image_matrix, title="Input Image")

    # TODO : Find coordinates of four corners of your inner Image ( X,Y format)
    #  Order of coordinates: Upper Left, Upper Right, Down Left, Down Right
    pts1 = np.float32([[107, 216], [363, 178], [159, 644], [490, 571]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = getPerspectiveTransform(pts1, pts2)

    warpedImage = warpPerspective(image_matrix, m, width, height)
    showWarpPerspective(warpedImage)

    grayScalePic = grayScaledFilter(warpedImage)
    showImage(grayScalePic, title="Gray Scaled")

    crazyImage = crazyFilter(warpedImage)
    showImage(crazyImage, title="Crazy Filter")

    customFilter(warpedImage)

    croppedImage = cropImg(warpedImage, 50, 300, 50, 225)
    showImage(croppedImage, title="Cropped Image")

    scaledImage = scaleImg(warpedImage, 2, 3)
    showImage(scaledImage, title="Scaled Image")
