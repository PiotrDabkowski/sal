''' Converts an image of an arbitrary size to a fixed sized image.
    Supports multiple modes and coordinate conversion between old and new image.'''
import cv2
import numpy as np



def multi_coord_support(f):
    def func(self, *xy):
        if len(xy)==1:
            xy = xy[0]
        assert len(xy)%2==0, 'Coords must be given in pairs xyxyxyxy...'
        res = ()
        for i in xrange(0, len(xy), 2):
            res += f(self, xy[i:i+2])
        return res
    return func


class BaseImageConverter:
    resulting_shape = [0, 0]
    original_shape = [0, 0]

    def get_resulting_img(self):
        raise NotImplementedError()

    @multi_coord_support
    def from_local_coords(self, xy):
        x, y = self._from_local_coords(xy)
        return min(max(x, 0), self.original_shape[1]-1), min(max(y, 0), self.original_shape[0]-1),

    @multi_coord_support
    def to_local_coords(self, xy):
        x, y = self._to_local_coords(xy)
        return min(max(x, 0), self.resulting_shape[1]-1), min(max(y, 0), self.resulting_shape[0]-1),

    @multi_coord_support
    def _from_local_coords(self, xy):
        raise NotImplementedError()

    @multi_coord_support
    def _to_local_coords(self, xy):
        raise NotImplementedError()


class ImageCropper(BaseImageConverter):
    def __init__(self, img, x_range, y_range):
        ''' Nice one because supports out of range crops. 
            Eg cropping 32x32 image using x_range (-10, 32) and the same y_range will result in 42x42 image with 
            original image in bottom right corner and the unknown parts set to 0 color. Assumes that you actually crop something...'''
        self.x_range = x_range
        self.y_range = y_range
        h, w, c = img.shape
        self.original_shape = h, w
        rh = y_range[1]-y_range[0]
        rw = x_range[1]-x_range[0]
        self.resulting_shape = rh, rw
        template = np.zeros((rh, rw, c), dtype=np.float32)
        self.real_x_range = map(lambda x: min(max(0, x), w), x_range)
        self.real_y_range = map(lambda y: min(max(0, y), h), y_range)
        self.template_x_offset = max(0, -x_range[0])
        self.template_y_offset = max(0, -y_range[0])
        template[
            self.template_y_offset : self.template_y_offset + self.real_y_range[1] - self.real_y_range[0],
            self.template_x_offset : self.template_x_offset + self.real_x_range[1] - self.real_x_range[0],
            :
        ] = img[
            self.real_y_range[0]:self.real_y_range[1],
            self.real_x_range[0]:self.real_x_range[1],
            :
        ]
        self.resulting_img = template

    def get_resulting_img(self):
        return self.resulting_img

    @multi_coord_support
    def _from_local_coords(self, xy):
        x, y = xy
        return x+self.x_range[0], y+self.y_range[0]

    @multi_coord_support
    def _to_local_coords(self, xy):
        x, y = xy
        return x - self.x_range[0], y - self.y_range[0]



class ImageResizer(BaseImageConverter):
    def __init__(self, img, new_size):
        '''Much simpler than the cropper, simply resizes, size obviously is (h, w)'''
        if type(new_size)==int:
            new_size = (new_size, new_size)
        h, w, c = img.shape
        self.original_shape = h, w
        self.original_image = img
        self.resulting_shape = new_size
        self.resulting_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        self.y_scale = float(h)/new_size[0]
        self.x_scale = float(w)/new_size[1]

    def get_resulting_img(self):
        return self.resulting_img

    @multi_coord_support
    def _from_local_coords(self, xy):
        x, y = xy
        return int(round(x * self.x_scale)), int(round(y * self.y_scale))

    @multi_coord_support
    def _to_local_coords(self, xy):
        x, y = xy
        return int(round(x / self.x_scale)), int(round(y / self.y_scale))



class FixedAspectRatioNoCropping:
    def __init__(self, img, desired_side_length):  #must be a square for now, easy to extend to rect
        h, w, c = img.shape
        short = min(w, h)
        long = max(w, h)
        off = (long-short)/2
        if w < h:
            self.i1 = ImageCropper(img, (-off, -off+long), (0, long))
        else:
            self.i1 = ImageCropper(img, (0, long), (-off, -off + long))
        self.i2 = ImageResizer(self.i1.get_resulting_img(), (desired_side_length, desired_side_length))
        self.resulting_img = self.i2.get_resulting_img()
        self.original_image = img

    def get_resulting_img(self):
        return self.resulting_img

    @multi_coord_support
    def from_local_coords(self, xy):
        return self.i1.from_local_coords(self.i2.from_local_coords(xy))

    @multi_coord_support
    def to_local_coords(self, xy):
        return self.i2.to_local_coords(self.i1.to_local_coords(xy))

