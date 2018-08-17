-- TODO: don't hardcode the write role
set role jakarta_smart_city_traffic_safety_write;

create table validation.cvat_frames_unique as (                            
with
	t1 as (
		with -- attach task to each box
        	t1 as
                (with   -- attach boxes to paths
                        t1 as
                                (with  -- attach path attributes to paths
                                        t1 as
                                                (select track_id, text as attr, value as attr_value from -- get track attributes
                                                        (select value, spec_id, track_id from validation.engine_objectpathattributeval) as t1
                                                                left join
                                                        (select id, "text" from validation.engine_attributespec) as t2
                                                                on
                                                        (t1.spec_id = t2.id))
                                select id, job_id, track_id, attr, attr_value, label from
                                        (select * from
                                                t1
                                                        join
                                                validation.engine_objectpath as t2
                                                        on
                                                (t2.id = t1.track_id)) as t1
                                                left join
                                        (select id as label_id, name as "label" from validation.engine_label) as t2
                                                using(label_id)),
                        t2 as
                                (with t1 as (
				with t1 as (
							with -- attach box attributes to boxes
                                        t1 as
                                                (select box_id, text as frame_attr, value as frame_attr_value from  -- get attributes for individual boxes
                                                        (select value, box_id, spec_id from validation.engine_trackedboxattributeval) as t1
                                                                left join
                                                        (select id, "text" from validation.engine_attributespec) as t2
                                                                on
                                                        (t1.spec_id = t2.id))
                                        select xtl, ytl, xbr, ybr, occluded, frame, outside, track_id, frame_attr, frame_attr_value from
                                                validation.engine_trackedbox as t2
                                                        left join
                                                t1
                                                        on
                                                (t2.id = t1.box_id)
                           	)
                           	select t1.*, t2.label_id, t2.job_id from
                           	t1 left join validation.engine_objectpath as t2 on t1.track_id=t2.id
            )
            select xtl, ytl, xbr, ybr, occluded, frame, outside, track_id, frame_attr, frame_attr_value, job_id, t2.name as label from
            t1 left join validation.engine_label as t2 on t1.label_id=t2.id)
                        select t2.job_id, t2.track_id, t2.label, frame, xtl, ytl, xbr, ybr, occluded, outside, attr, attr_value, frame_attr, frame_attr_value from
                        --select job_id, t1.frame, id, label, xtl, ytl, xbr, ybr, occluded, outside, attr, attr_value, frame_attr, frame_attr_value from
                                t1
                                        full outer join
                                t2
                                        on
                                (t1.id = t2.track_id)
                                        order by
                                job_id, id, frame),
        t2 as (select job_id, name, path from
                                (select job_id, task_id from
                                        (select id as job_id, segment_id from validation.engine_job) as t1
                                                left join
                                        (select * from validation.engine_segment) as t2
                                                on
                                        (t1.segment_id = t2.id)) as t1
                                        left join
                                (select * from validation.engine_task) as t2
                                        on
                                (t1.task_id = t2.id))
        select name, path, track_id, label, frame, xtl, ytl, xbr, ybr, occluded, outside, attr, attr_value, frame_attr, frame_attr_value from
                t1
                        left join
                t2
                        using(job_id)
    )
    select name, path, track_id, label, frame, xtl, ytl, xbr, ybr, occluded, outside,
    -- json_agg(json_build_object((regexp_split_to_array((regexp_split_to_array(attr, E'='))[2], E':'))[1], attr_value)) as path_attr,
    --json_agg(case when frame_attr is null
    --	then null
    --	else json_build_object((regexp_split_to_array((regexp_split_to_array(frame_attr, E'='))[2], E':'))[1], frame_attr_value)
    --end) as frame_attr
    coalesce(json_agg(json_build_object((regexp_split_to_array((regexp_split_to_array(attr, E'='))[2], E':'))[1], attr_value)) filter
     (where attr is not null), null) as path_attr,
    coalesce(json_agg(json_build_object((regexp_split_to_array((regexp_split_to_array(frame_attr, E'='))[2], E':'))[1], frame_attr_value)) filter
     (where frame_attr is not null), null) as frame_attr
    from t1
    group by name, path, track_id, label, frame, xtl, ytl, xbr, ybr, occluded, outside
   );
     
create table validation.cvat_frames_interpolated as (
with foo_allinfo as (
        with foo_frameinfo as (
                with foo_complete as (
                        select *
                        from(with t1 as (select name as interp_name, generate_series(min(frame), max(frame)) as interp_frame
                        		from validation.cvat_frames_unique
                        		group by name),
							t2 as (select name, track_id from validation.cvat_frames_unique
								group by name, track_id)
                      	select interp_name, t2.track_id as interp_track_id, interp_frame from
                     	t1 left join t2 on t1.interp_name=t2.name) frames
                        left join validation.cvat_frames_unique on
                        (cvat_frames_unique.frame=frames.interp_frame and
                        cvat_frames_unique.name=frames.interp_name and
                        cvat_frames_unique.track_id=frames.interp_track_id)
                )
                select *,
                max(interp_frame) filter (where xtl is not null) over (partition by interp_name, interp_track_id order by interp_frame rows between unbounded preceding and 1 preceding) as f_prev,
                min(interp_frame) filter (where xtl is not null) over (partition by interp_name, interp_track_id order by interp_frame rows between 1 following and unbounded following) as f_next,
                (xtl is null) as interpolated
                from foo_complete
        )
        select A.interp_frame as frame,
        coalesce(A.interp_name, B.name) as name,
        coalesce(A.path, B.path) as path,
        coalesce(A.interp_track_id, B.track_id) as track_id,
        coalesce(A.label, B.label) as label,
        A.xtl, A.ytl, A.xbr, A.ybr,
        coalesce(A.outside, B.outside) as outside,
        coalesce(A.occluded, B.occluded) as occluded,
        coalesce(A.path_attr, B.path_attr) as path_attr,
        coalesce(A.frame_attr, B.frame_attr) as frame_attr,
        A.f_prev, A.f_next, A.interpolated,
        B.xtl as xtl_prev, B.ytl as ytl_prev, B.xbr as xbr_prev, B.ybr as ybr_prev,
        C.xtl as xtl_next, C.ytl as ytl_next, C.xbr as xbr_next, C.ybr as ybr_next,
        A.f_next - A.f_prev as denominator,
        A.interp_frame - A.f_prev as weight_next,
        A.f_next - A.interp_frame as weight_prev
        from foo_frameinfo A
        left join validation.cvat_frames_unique B
        on (A.f_prev = B.frame and A.interp_track_id=B.track_id)
        left join validation.cvat_frames_unique C
        on (A.f_next = C.frame and A.interp_track_id=C.track_id)
)
select frame+1 as frame, name, track_id, label, outside, occluded, path_attr, frame_attr, interpolated,
coalesce(xtl, (weight_prev*xtl_prev + weight_next*xtl_next)/denominator, xtl_prev) as xtl,
coalesce(ytl, (weight_prev*ytl_prev + weight_next*ytl_next)/denominator, ytl_prev) as ytl,
coalesce(xbr, (weight_prev*xbr_prev + weight_next*xbr_next)/denominator, xbr_prev) as xbr,
coalesce(ybr, (weight_prev*ybr_prev + weight_next*ybr_next)/denominator, ybr_prev) as ybr
from foo_allinfo
where (outside is false)
);

create table validation.cvat_frames_interpmotion as (
with t1 as (
	with t1 as (
		select *,(xtl+xbr)/2 as xc,(ytl+ybr)/2 as yc,
		max(frame) over (partition by track_id order by frame rows between 1 preceding and 1 preceding) as frame_prev
		from validation.cvat_frames_interpolated
	)
	select t1.*, t2.xc as xc_prev, t2.yc as yc_prev
	from t1 left join t1 t2
	on t1.frame_prev=t2.frame and t1.track_id=t2.track_id
	order by track_id, frame
)
select frame, name, track_id, label, occluded, path_attr, frame_attr, interpolated, xtl, ytl, xbr, ybr,
xc-xc_prev as deltax, yc-yc_prev as deltay from t1
);