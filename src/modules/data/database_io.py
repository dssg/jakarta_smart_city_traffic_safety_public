# ============ Base imports ======================
from io import StringIO
from functools import partial
# ====== External package imports ================
import numpy as np
import psycopg2 as psy
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class DatabaseIO:
    """Class which mediates all interactions with the database
    """
    def __init__(self, testing=False):
        """Defines the schemas, and creates psycopg2 connection to the database

        Note: the schema in this class should always match that in the database

        :param testing: boolean, if True, nothing is written to the database, it just prints the commands which will be run
        """
        self.write_role = conf.db.write_role
        self.conn = psy.connect(
            database=conf.db.db_name,
            user=conf.db.user,
            password=conf.db.pw,
            host=conf.db.host,
            port=conf.db.port,
        )
        self.testing = testing
        #TODO: update the schemas to include annotation tables
        self.schemas = {"raw":
                            {"cameras":
                                ((
                                     "id",
                                     "site_name",
                                     "object_id",
                                     "cctv_id",
                                     "location",
                                     "kota",
                                     "kota_en",
                                     "kecamatan",
                                     "kelurahan",
                                     "versi",
                                     "lattitude",
                                     "longitude",
                                     "url",
                                     "fps",
                                     "height",
                                     "width"
                                     ),
                                     (
                                     "uuid",
                                     "varchar(64)",
                                     "integer",
                                     "integer",
                                     "varchar(64)",
                                     "varchar(32)",
                                     "varchar(32)",
                                     "varchar(32)",
                                     "varchar(32)",
                                     "varchar(16)",
                                     "float",
                                     "float",
                                     "varchar(128)",
                                     "integer",
                                     "integer",
                                     "integer"
                                     )),
                             "video_metadata":
                                ((
                                     "id",
                                     "file_md5_chunk_7mb",
                                     "file_name",
                                     "camera_id",
                                     "time_start_subtitles",
                                     "time_end_subtitles",
                                     "file_location",
                                     "file_path"
                                     ),
                                     (
                                     "uuid",
                                     "varchar(32)",
                                     "varchar(256)",
                                     "uuid",
                                     "varchar(32)",
                                     "varchar(32)",
                                     "varchar(8)",
                                     "varchar(256)"
                                     )),
                             "subtitles":
                                 ((
                                      "video_id",
                                      "subtitle_number",
                                      "display_time_start",
                                      "display_time_end",
                                      "subtitle_text"
                                      ),
                                      (
                                      "uuid",
                                      "integer",
                                      "varchar(16)",
                                      "varchar(16)",
                                      "varchar(32)"
                                      )),
                             "packet_stats":
                                 ((
                                      "video_id",
                                      "pts_time",
                                      "dts_time",
                                      "size",
                                      "pos",
                                      "flags"
                                      ),
                                      (
                                      "uuid",
                                      "varchar(16)",
                                      "varchar(16)",
                                      "varchar(16)",
                                      "varchar(16)",
                                      "varchar(16)"
                                      )),
                             "frame_stats":
                                 ((
                                      "video_id",
                                      "key_frame",
                                      "pkt_pts_time",
                                      "pkt_dts_time",
                                      "best_effort_timestamp_time",
                                      "pkt_size",
                                      "pict_type",
                                      "coded_picture_number",
                                      ),
                                      (
                                      "uuid",
                                      "varchar(16)",
                                      "varchar(16)",
                                      "varchar(16)",
                                      "varchar(16)",
                                      "integer",
                                      "char(1)",
                                      "integer"
                                      ))
                            },
                         "main": {
                                "db_failures":
                                    ((
                                        "time",
                                        "description"
                                    ),
                                    (
                                        "varchar(32)",
                                        "json"
                                    ))
                            },
                         "results": {
                                "models":
                                    ((
                                        "model_number",
                                        "pipeline_config",
                                        "datetime_created",
                                    ),(
                                        "integer",
                                        "json",
                                        "timestamp NULL DEFAULT Now()",
                                    )),
                                "box_motion":
                                    ((
                                        "model_number",
                                        "video_id",
                                        "video_file_name",
                                        "frame_number",
                                        "mean_x",
                                        "mean_y",
                                        "mean_delta_x",
                                        "mean_delta_y",
                                        "magnitude",
                                        "angle_from_vertical",
                                        "box_id",
                                        "datetime_created",
                                     ),(
                                        "integer",
                                        "uuid",
                                        "varchar(256)",
                                        "integer",
                                        "double precision",
                                        "double precision",
                                        "double precision",
                                        "double precision",
                                        "double precision",
                                        "double precision",
                                        "integer",
                                        "timestamp NULL DEFAULT Now()",
                                    )),
                                "boxes":
                                    ((
                                        "model_number",
                                        "video_id",
                                        "video_file_name",
                                        "frame_number",
                                        "xtl",
                                        "ytl",
                                        "xbr",
                                        "ybr",
                                        "objectness",
                                        "pedestrian",
                                        "bicycle",
                                        "car",
                                        "motorbike",
                                        "bus",
                                        "train",
                                        "truck",
                                        "semantic_segment_bottom_edge_mode",
                                        "box_id",
                                        "datetime_created",
                                     ),
                                     (
                                         "integer",
                                         "uuid",
                                         "varchar(256)",
                                         "int4",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "float8",
                                         "varchar(32)",
                                         "integer",
                                         "timestamp NULL DEFAULT Now()",
                                     )),
                                "frame_stats":
                                    ((
                                         "model_number",
                                         "video_id",
                                         "video_file_name",
                                         "frame_number",
                                         "pedestrian_counts",
                                         "bicycle_counts",
                                         "car_counts",
                                         "motorbike_counts",
                                         "bus_counts",
                                         "train_counts",
                                         "truck_counts",
                                         "pedestrian_sums",
                                         "bicycle_sums",
                                         "car_sums",
                                         "motorbike_sums",
                                         "bus_sums",
                                         "train_sums",
                                         "truck_sums",
                                         "datetime_created",
                                     ),
                                     (
                                         "integer",
                                         "uuid",
                                         "varchar(256)",
                                         "integer",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "double precision",
                                         "timestamp NULL DEFAULT Now()",
                                     ))
                         }
                        }

    class RunSql(object):
        """Decorator class which wraps sql-generating functions and runs generic sql
        """
        def __init__(self, decorated):
            """ Stores function to be decorated

            :param decorated:  function to be decorated
            """
            self.decorated = decorated

        def __call__(self, dbio, *args, **kwargs):
            """called when the decorated function is called

            :param dbio: database connection to use for connecting
            :param args: arguments to be passed to the decorated function
            :param kwargs: keyword arguments to be passed to the decorated function
            :return: None
            """
            sql = self.decorated(dbio, *args, **kwargs)
            if not dbio.testing:
                logger.debug("'execute' will run\n{}".format(sql))
                cur = dbio.conn.cursor()
                cur.execute(sql)
                cur.close()
                dbio.conn.commit()
            else:
                logger.info("'execute' will run\n{}".format(sql))

        def __get__(self, dbio, owner):
            """I'm not 100% sure about the need for this, something about being a decorator defined within a class in
            order to have access to the parent class instance
            """
            return partial(self.__call__, dbio)

    class RunSqlSelect(object):
        """Decorator class which wraps sql-generating functions and runs a select statement and returns result of the select statement
        """
        def __init__(self, decorated):
            """ Stores function to be decorated

            :param decorated:  function to be decorated
            """
            self.decorated = decorated

        def __call__(self, dbio, *args, **kwargs):
            """called when the decorated function is called

            :param dbio: database connection to use for connecting
            :param args: arguments to be passed to the decorated function
            :param kwargs: keyword arguments to be passed to the decorated function
            :return: results: a tuple of tuples, where each inner tuple contains the values for a row returned by this
            query; columns: tuple containing column names
            """
            sql = self.decorated(dbio, *args, **kwargs)
            if not dbio.testing:
                logger.debug(f"running select:{sql}")
                cur = dbio.conn.cursor()
                cur.execute(sql)
                results = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                cur.close()
                dbio.conn.commit()
                return results, columns
            else:
                logger.debug("will run:{sql}")
                return None, None

        def __get__(self, dbio, owner):
            """I'm not 100% sure about the need for this, something about being a decorator defined within a class in
            order to have access to the parent class instance
            """
            return partial(self.__call__, dbio)

    """Decorator class which wraps sql-generating functions and runs them using the copy expert function
    
    Good for copying large volumnes of data to the database
    """
    class CopyExpert(object):
        def __init__(self, decorated):
            """ Stores function to be decorated

            :param decorated:  function to be decorated
            """
            self.decorated = decorated

        def __call__(self, dbio, *args, **kwargs):
            """called when the decorated function is called

            :param dbio: database connection to use for connecting
            :param args: arguments to be passed to the decorated function
            :param kwargs: keyword arguments to be passed to the decorated function
            :return: None
            """
            sql, f = self.decorated(dbio, *args, **kwargs)
            if not dbio.testing:
                logger.debug("'copy_expert' will run\n{}".format(sql))
                cur = dbio.conn.cursor()
                cur.copy_expert(sql, f)
                cur.close()
                dbio.conn.commit()
                f.close()
            else:
                logger.info("'copy_expert' will run\n{}".format(sql))
                f.close()

        def __get__(self, dbio, owner):
            """I'm not 100% sure about the need for this, something about being a decorator defined within a class in
            order to have access to the parent class instance
            """
            return partial(self.__call__, dbio)

    @RunSql
    def create_schema(self, schema):
        """ create a schema in the database

        :param schema: name of schema
        :return: sql statement that gets run on the database
        """
        sql = f'set role {self.write_role}; ' \
              + f'CREATE SCHEMA IF NOT EXISTS {schema};'
        return sql

    @RunSql
    def create_table(self, schema, table):
        """ create a table in the database

        :param schema: name of schema which will contain the table
        :param table: name of table to be created
        :return: sql statement that gets run on the database
        """
        fields = ", ".join([" ".join(t) for t in zip(self.schemas[schema][table][0], self.schemas[schema][table][1])])
        sql = f'set role {self.write_role}; '  \
            + f'CREATE TABLE IF NOT EXISTS {schema}.{table} ( {fields} );'
        return sql

    @RunSql
    def drop_table(self, schema, table):
        """ drop a table from the database

        :param schema: name of schema which contains the table to be dropped
        :param table: name of the table to be dropped
        :return: sql statement that gets run on the database
        """
        sql = f'set role {self.write_role}; ' \
            + f'DROP TABLE IF EXISTS {schema}.{table};'
        return sql

    @RunSql
    def drop_schema(self, schema):
        """ drop a schema from the database

        :param schema: name of schema to be dropped
        :return: sql statement that gets run on the database
        """
        sql = f'set role {self.write_role}; ' \
              + f'DROP SCHEMA IF EXISTS {schema};'
        return sql

    @CopyExpert
    def copy_file_to_table(self, schema, table, filepath):
        """ copy a file from a filepath to a table in the database

        Note: schema, table, and table fields come from this class, and they should match what's in the database as well
        as what's in the file

        :param schema: name of schema containing the relevant table
        :param table: name of table which will contain the information
        :return: sql statement that gets run on the database
        """
        fields = ", ".join(self.schemas[schema][table][0])
        sql = f'set role {self.write_role}; ' \
              f'COPY {schema}.{table}( {fields} ) FROM stdin WITH DELIMITER \',\' CSV header;'
        return sql, open(filepath, 'r')

    @CopyExpert
    def copy_np_array_to_table(self, schema, table, a):
        """ copy a numpy array to a table in the database

        Note: schema, table, and table fields come from this class, and they should match what's in the database as well
        as what's in the file

        :param schema: name of schema containing the relevant table
        :param table: name of table which will contain the information
        :param a: array to be copied
        :return: sql statement that gets run on the database
        """
        fields = ", ".join(self.schemas[schema][table][0])
        sql = f'set role {self.write_role}; ' \
              f'COPY {schema}.{table}( {fields} ) FROM stdin WITH DELIMITER \',\' CSV header;'
        return sql, StringIO(np.array2string(a, separator=","))

    @CopyExpert
    def copy_string_to_table(self, schema, table, s, separator=","):
        """ copy a generic string to a table in the database

        Note: schema, table, and table fields come from this class, and they should match what's in the database as well
        as what's in the file

        :param schema: name of schema containing the relevant table
        :param table: name of table which will contain the information
        :param s: string to be copied
        :param separator: delimiter used in this string
        :return: sql statement that gets run on the database
        """
        #fields = (separator + " ").join(self.schemas[schema][table][0])
        fields = s.split("\n")[0].replace(separator, ",")
        sql = f'set role {self.write_role}; ' \
              f'COPY {schema}.{table}( {fields} ) FROM stdin WITH DELIMITER \'{separator}\' CSV header;'
        return sql, StringIO(s)

    @RunSql
    def insert_into_table(self, schema, table, fields, values):
        """inserts a single row into a table

        :param schema: name of schema contianing the table to be written to
        :param table: name of table to be written to
        :param fields: iterable, column names which match the database and the columns
        :param values: iterable, values to be inserted for the specified column names
        :return: sql statement that gets run on the database
        """
        sql = f'set role {self.write_role}; ' \
              f'INSERT INTO {schema}.{table} ( {", ".join(fields)} ) VALUES ( {", ".join(values)} );'
        return sql

    @RunSqlSelect
    def get_camera_id(self, camera_name):
        """Get the id of a particular camera from the cameras table

        Note: schema and tables are hardcoded, so if that changes, this should also change

        :param camera_name: name of camera for which you want the id
        :return: sql statement which gets run on the database
        """
        sql = 'set role {}; '.format(self.write_role) \
            + f"SELECT id, site_name FROM raw.cameras WHERE site_name = '{camera_name}'"
        return sql

    @RunSqlSelect
    def _get_video_info(self, file_name):
        """get the information for a video by matching its file name to the ones in the raw.video_metadata table

        Note: schema and table are hardcoded, so if that changes, this should also change

        :param file_name: name to match against files in the table
        :return:sql statement which gets run on the database
        """
        sql = f"set role {self.write_role}; "\
              + "select * from "\
              + f"(select * from raw.video_metadata where file_name like '{file_name}') as vid "\
              + "left join "\
              + "raw.cameras as cams "\
              + "on "\
              + "(cams.id = vid.camera_id); "
        return sql

    def get_video_info(self, file_name):
        """gets video information and returns it as a dictionary

        :param file_name: name to match against files in the metadata_table
        :return: dictionary containing key value pairs with column names as keys and row values as value
        """
        info, colnames = self._get_video_info(file_name)
        if info is None or len(info)==0:
            return None
        return dict(zip(colnames, info[0]))

    @RunSqlSelect
    def get_video_annotations(self, file_name):
        """Gets video annotation data from the database

        Note: schema and table are hard coded, so if that changes, this should change as well

        :param file_name: name of file for which to retrieve video annotations
        :return: sql statement which gets run on the database
        """
        sql = f"SET role {self.write_role}; " \
            + f"SELECT * FROM validation.cvat_frames_interpmotion " \
            + f"WHERE name = '{file_name}'; "
        return sql

    @RunSqlSelect
    def get_results_boxes(self, file_name, model_no):
        """Gets all boxes from the results.boxes table

        Note: schema and table are hard coded, so if that changes, this should change as well
        
        :param file_name: to match in the results boxes table
        :param model_no: the model number to get results for
        :return: sql statement which gets run on the database
        """
        sql = f"SET role {self.write_role}; " \
            + f"SELECT * FROM results.boxes " \
            + f"WHERE video_file_name = '{file_name}'' and model_number = '{model_no}'"
        return sql
    
    @RunSqlSelect
    def get_results_motion(self, file_name, model_no):
        sql = f"SET role {self.write_role}; " \
            + f"WITH foo as (" \
            + f"SELECT * FROM results.boxes " \
            + f"WHERE video_file_name = '{file_name}' and model_number = '{model_no}')" \
            + f"SELECT foo.*, " \
            + f"results.box_motion.mean_delta_x, results.box_motion.mean_delta_y, results.box_motion.magnitude " \
            + f"from foo " \
            + f"LEFT JOIN results.box_motion " \
            + f"ON foo.box_id=results.box_motion.box_id and foo.model_number=results.box_motion.model_number;"
        return sql

    @RunSql
    def upload_semantic_segments_to_boxes(self, data):
        """script to upload semantic segment information to boxes

        Note: schema and table are hard coded, so if that changes, this should change as well

        :param data: tuple of strings to be written to the table
        :return: sql statement which gets run on the database
        """
        #data_str = 'array["' + '","'.join(data) + '"]'
        data_str = "array['" + "','".join(data) + "']"
        sql = f"SET role {self.write_role}; " \
              + f"update results.boxes " \
              + f"set semantic_segment_bottom_edge_mode = ({data_str})[id];"
        return sql

    @RunSqlSelect
    def get_max_model_number(self):
        return f"SET role {self.write_role}; Select max(models.model_number) from results.models;"

    def create_all_schemas_and_tables(self):
        """function which creates all schemas and tables specifed by this class in self.schemas

        """
        for schema, tables in self.schemas.items():
            self.create_schema(schema)
            for table in tables.keys():
                self.create_table(schema, table)


if __name__ == "__main__":
    dbio = DatabaseIO()
    import pdb; pdb.set_trace()
