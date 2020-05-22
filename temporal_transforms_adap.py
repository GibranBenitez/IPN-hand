import random
import math


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):

        rand_end = max(0, len(frame_indices) - self.size - 1)
        if rand_end > 0:
            out = frame_indices[:self.size]
        else:
            out = []
            extra_pad = self.size % len(frame_indices)
            repi = [self.size//len(frame_indices)] * len(frame_indices)

            for index in range(len(frame_indices)):
                buf = 1 if extra_pad > 0 else 0
                extra_pad -= 1
                for i in range(repi[index]+buf):
                    out.append(frame_indices[index])
        # for index in out:
        #     if len(out) >= self.size:
        #         break
        #     out.append(index)


        return out



class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        if rand_end > 0:
            center_index = len(frame_indices) // 2
            begin_index = max(0, center_index - (self.size // 2))
            end_index = min(begin_index + self.size, len(frame_indices))

            out = frame_indices[begin_index:end_index]
        else:
            out = []
            extra_pad = self.size % len(frame_indices)
            repi = [self.size//len(frame_indices)] * len(frame_indices)

            for index in range(len(frame_indices)):
                buf = 1 if extra_pad > 0 else 0
                extra_pad -= 1
                for i in range(repi[index]+buf):
                    out.append(frame_indices[index])
        # for index in out:
        #     if len(out) >= self.size:
        #         break
        #     out.append(index)


        return out





class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)

        if rand_end > 0:
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.size, len(frame_indices))

            out = frame_indices[begin_index:end_index]
        else:
            out = []
            extra_pad = self.size % len(frame_indices)
            repi = [self.size//len(frame_indices)] * len(frame_indices)

            for index in range(len(frame_indices)):
                buf = 1 if extra_pad > 0 else 0
                extra_pad -= 1
                for i in range(repi[index]+buf):
                    out.append(frame_indices[index])
        # for index in out:
        #     if len(out) >= self.size:
        #         break
        #     out.append(index)


        return out



class TemporalPadRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, frame_list):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        begin_t = frame_list[0] - self.pad
        end_t = frame_list[-1] + self.pad
        frame_indices = list(range(begin_t, end_t + 1))
        rand_end = max(0, len(frame_indices) - self.size - 1)

        if rand_end > 0:
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.size, len(frame_indices))

            out = frame_indices[begin_index:end_index]
        else:
            out = []
            extra_pad = self.size % len(frame_indices)
            repi = [self.size//len(frame_indices)] * len(frame_indices)

            for index in range(len(frame_indices)):
                buf = 1 if extra_pad > 0 else 0
                extra_pad -= 1
                for i in range(repi[index]+buf):
                    out.append(frame_indices[index])
        # for index in out:
        #     if len(out) >= self.size:
        #         break
        #     out.append(index)


        return out