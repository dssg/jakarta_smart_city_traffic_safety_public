# ============ Base imports ======================
import os
from glob import glob
from uuid import uuid4
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.utils.os import syscall_decode as sp
from src.modules.data.database_io import DatabaseIO as DBIO
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class VideoFile:
    """
    Class which houses interactions with a video file, including extracting metadata, and uploading it to the database
    """
    def __init__(self, path=None, get_info=True, dbio=None, aws=None):
        """stores params, optionally extracts information about video streams, and determines if metadata has been extracted

        :param path: location of this filesystem on the storage system
        :param get_info: if True, then attempts to extract video stream information
        :param dbio: DatabaseIO instance
        :param aws: AwsIO instance
        """
        # where this class gets/puts data:
        if path is not None:
            self.set_paths(path)

        # run ffprobe to get description of streams
        self.stream_descriptions = None
        self.format = None
        if get_info:
            self.get_streams_and_format()

        # keep track of actions which take a long time
        try:
            self._frames_extracted = True if os.path.exists(self.frame_stats_path) else False
        except FileNotFoundError:
            self._frames_extracted = False
        try:
            self._packets_extracted = True if os.path.exists(self.packet_stats_path) else False
        except FileNotFoundError:
            self._packets_extracted = False
        try:
            self._subtitles_extracted = True if os.path.exists(self.subtitles_path) else False
        except FileNotFoundError:
            self._subtitles_extracted = False
        try:
            self._errors_checked = True if os.path.exists(self.errors_path) else False
        except FileNotFoundError:
            self._errors_checked = False

        # set database connection
        self._dbio = dbio
        self._aws = aws

    def set_paths(self, path):
        self.path = path
        self.basename = os.path.basename(path)
        self.name, self.extension = self.basename.split(".")
        self.frame_stats_path = os.path.join(conf.dirs.frame_stats, self.name + ".csv")
        self.packet_stats_path = os.path.join(conf.dirs.packet_stats, self.name + ".csv")
        self.raw_subtitles_path = os.path.join(conf.dirs.subtitles, self.name + ".srt")
        self.subtitles_path = os.path.join(conf.dirs.subtitles, self.name + ".csv")
        self.errors_path = os.path.join(conf.dirs.errors, self.name + ".csv")
        self.hash_path = os.path.join("")

    @property
    def dbio(self):
        if self._dbio is None:
            self._dbio = DBIO()
        return self._dbio

    @property
    def aws(self):
        if self._aws is None:
            self._aws = DBIO()
        return self._aws


    def get_streams_and_format(self):
        stdout, stderr, returnstatus = sp(["ffprobe", "-v", "error", "-show_format", "-show_streams", self.path])
        streams = []  # list of all streams in the video (each is a dictionary of settings)
        _format = {}  # dictionary of video format settings
        in_stream = False
        in_format = False
        stream = None
        if returnstatus == 0:
            for line in stdout.split('\n'):
                if "[STREAM]" in line:
                    in_stream = True
                    stream = {}
                elif "[/STREAM]" in line:
                    streams.append(stream)
                    in_stream = False
                elif in_stream:
                    attribute, value = line.split("=")
                    stream[attribute] = value
                elif "[FORMAT]" in line:
                    in_format = True
                elif "[/FORMAT]" in line:
                    self.format = _format
                    in_format = False
                elif in_format:
                    attribute, value = line.split("=")
                    _format[attribute] = value
                else:
                    continue
            self.stream_descriptions = streams
        return returnstatus

    # Note: entries order not guaranteed.  Instead the usual display order will be retained, so you should match that.
    # in order for the field headers to match the data in the output csv file
    def extract_frame_stats(self,
                            entries="key_frame,pkt_pts_time,pkt_dts_time,best_effort_timestamp_time,pkt_size,pict_type,"
                                    + "coded_picture_number", forcenew=False):
        if self._frames_extracted and (not forcenew):
            logger.debug("frames already extracted")
            return 0
        if not os.path.exists(conf.dirs.frame_stats):
            os.makedirs(conf.dirs.frame_stats)
        # Note: Tried redirecting subprocess stdout directly to file but it was ~20s slower.
        stdout, stderr, returnstatus = sp(['ffprobe', "-loglevel", "panic", "-of", "csv=p=0", "-select_streams", "v",
                                           "-show_frames", "-show_entries", "frame=" + entries, self.path])
        if returnstatus == 0:
            with open(self.frame_stats_path, 'w') as f:
                f.write(entries + "\n")
                f.write(stdout)
            self._frames_extracted = True
        return returnstatus

    def get_frame_stats(self, forcenew=False):
        returnstatus = 0
        if (not self._frames_extracted) or forcenew:
            returnstatus = self.extract_frame_stats(forcenew=forcenew)
        if returnstatus == 0:
            with open(self.frame_stats_path) as f:
                return f.read()
        return ""

    def extract_packet_stats(self, entries="pts_time,dts_time,size,pos,flags", forcenew=False):
        if self._packets_extracted and (not forcenew):
            return 0
        if not os.path.exists(conf.dirs.packet_stats):
            os.makedirs(conf.dirs.packet_stats)
        stdout, stderr, returnstatus = sp(['ffprobe', "-loglevel", "panic", "-of", "csv=p=0", "-select_streams", "v",
                                           "-show_packets", "-show_entries", "packet=" + entries, self.path])
        if returnstatus == 0:
            with open(self.packet_stats_path, 'w') as f:
                f.write(entries + "\n")
                f.write(stdout)
            self._packets_extracted = True
        return returnstatus

    def get_packet_stats(self, forcenew=False):
        returnstatus = 0
        if (not self._packets_extracted) or forcenew:
            returnstatus = self.extract_packet_stats(forcenew=forcenew)
        if returnstatus == 0:
            with open(self.packet_stats_path) as f:
                return f.read()
        return ""

    def extract_subtitles(self, forcenew=False):
        if self._subtitles_extracted and (not forcenew):
            return 0
        if not os.path.exists(conf.dirs.subtitles):
            os.makedirs(conf.dirs.subtitles)
        stdout, stderr, returnstatus = sp(['ffmpeg', '-y', "-i", self.path, '-f', 'srt', "-map", "0:0", "-vsync", "0",
                                           self.raw_subtitles_path])
        if returnstatus == 0:
            with open(self.subtitles_path, 'w') as f:
                f.write("subtitle_number,start_time,end_time,subtitle\n")
                sub_number = start_time = end_time = subtitle = ""
                with open(self.raw_subtitles_path, 'r') as f2:
                    lines = f2.read()
                for line in lines.split("\n"):
                    if " --> " in line:
                        start_time, end_time = line.replace(",", ".").split(" --> ")
                    elif "AM" in line or "PM" in line:
                        subtitle = line.replace(",", "").replace("\n", "")
                    else:
                        try:
                            sub_number = int(line)
                        except ValueError:  # blank line
                            f.write("{},{},{},{}\n".format(sub_number, start_time, end_time, subtitle))
                            sub_number = start_time = end_time = subtitle = ""
            self._subtitles_extracted = True
        return returnstatus

    def get_subtitles(self, forcenew=False):
        returnstatus = 0
        if (not self._subtitles_extracted) or forcenew:
            returnstatus = self.extract_subtitles(forcenew=forcenew)
        if returnstatus == 0:
            with open(self.subtitles_path) as f:
                return f.read()
        return ""

    def check_for_errors(self, etypes=('Tuncating packet', 'non monotonically increasing dts', 'Read error', 'no frame',
                                       'SPS decoding failure'), forcenew=False):
        if self._errors_checked and (not forcenew):
            return 0
        if not os.path.exists(conf.dirs.errors):
            os.makedirs(conf.dirs.errors)
        stdout, stderr, returnstatus = sp(['ffmpeg', '-v', 'error', '-i', self.path, '-vsync', '0', '-f', 'null', '-'])
        if returnstatus == 0:
            counts = [0] * len(etypes)
            for line in stderr.split("\n"):
                for i, etype in enumerate(etypes):
                    if etype in line:
                        counts[i] += 1
                        continue
            with open(self.errors_path, "w") as f:
                f.write(",".join(etypes) + "\n" + ",".join([str(cnt) for cnt in counts]))
            self._errors_checked = True
        return returnstatus

    def get_errors(self, forcenew=False):
        returnstatus = 0
        if (not self._errors_checked) or forcenew:
            returnstatus = self.check_for_errors(forcenew=forcenew)
        if returnstatus == 0:
            with open(self.errors_path) as f:
                return f.read()
        return ""

    @staticmethod
    def url_to_filename(url):
        """Returns our filename from their url."""
        split_url = url.split('.')[4].split('%')
        part = url.split('.')[5]
        if part[:4] != 'part':
            part = 'part0'
        partial_name = split_url[0] + '_' + split_url[1][2:] + '_' + split_url[2][2:] + '_' + split_url[4][2:]
        file_name = partial_name + '_' + part + '.mkv'
        return file_name

    def upload_vid_metadata_to_db(self, file_loc="fs"):
        # Get metadata information
        uuid = f"'{uuid4()!s}'"
        # generate a hash for this file
        try:
            stdout, stderr, returncode = sp([os.path.join(conf.dirs.scripts, "s3etag.sh"), self.path, "7"])
            hash = stdout[:32]
        except:
            hash = -1
        file_md5_chunk_7mb = f"'{hash!s}'"
        if "Gambir" not in self.basename:
            camera_name = "_".join(self.basename.split("_")[:2])
        else:
            camera_name = "_".join(("-".join(self.basename.split("-")[:3]), self.basename.split("-")[3][0]))
        camera_id = f"'{self.dbio.get_camera_id(camera_name)[0][0][0]}'"
        subs = self.get_subtitles().split("\n")
        time_start_subtitles = f"'{subs[1].split(',')[3]}'"
        time_end_subtitles = f"'{subs[-3].split(',')[3]}'"
        file_location = f"'{file_loc}'"
        file_path = f"'{self.path}'"
        file_name = f"'{self.basename}'"

        # frame stats
        # prepend uuid to file
        logger.debug("creating uuid column in frame stats file")
        new_frame_stats_file = f'{conf.dirs.frame_stats}{self.name}_uuid.csv'
        sp(['sed', f's/^/{uuid[1:-1]},/', self.frame_stats_path], stdout=open(new_frame_stats_file, 'w'), stderr=None)

        # packet stats
        # prepend uuid to file
        logger.debug("creating uuid column in packet stats file")
        new_packet_stats_file = f'{conf.dirs.packet_stats}{self.name}_uuid.csv'
        sp(['sed', f's/^/{uuid[1:-1]},/', self.packet_stats_path], stdout=open(new_packet_stats_file, 'w'), stderr=None)

        # subtitles
        # prepend uuid to file
        logger.debug("creating uuid column in subtitles file")
        new_subtitles_file = f'{conf.dirs.subtitles}{self.name}_uuid.csv'
        sp(['sed', f's/^/{uuid[1:-1]},/', self.subtitles_path], stdout=open(new_subtitles_file, 'w'), stderr=None)


        # copy to db
        self.dbio.insert_into_table("raw", "video_metadata", ("id", "file_md5_chunk_7mb", "file_name", "camera_id",
                                                               "time_start_subtitles", "time_end_subtitles",
                                                               "file_location", "file_path"),
                                    (uuid, file_md5_chunk_7mb, file_name, camera_id,
                                     time_start_subtitles, time_end_subtitles,
                                     file_location, file_path))
        self.dbio.copy_file_to_table("raw", "frame_stats", new_frame_stats_file)
        self.dbio.copy_file_to_table("raw", "subtitles", new_subtitles_file)
        self.dbio.copy_file_to_table("raw", "packet_stats", new_packet_stats_file)


if __name__ == "__main__":
    setup("video_file")
    vid_file = glob(os.path.join(conf.dirs.videos, "*.mkv"))[0]
    vid = VideoFile(path=vid_file, dbio=DBIO(testing=False))
    vid.upload_vid_metadata_to_db()
