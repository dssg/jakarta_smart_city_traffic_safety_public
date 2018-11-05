set role jakarta_smart_city_traffic_safety_write;

--=======================================================================
select value, box_id, spec_id from public.engine_trackedboxattributeval;
select id, "text" from public.engine_attributespec;
 
select box_id, text, value from  -- get attributes for individual boxes
	(select value, box_id, spec_id from public.engine_trackedboxattributeval) as t1
		left join
	(select id, "text" from public.engine_attributespec) as t2
		on
	(t1.spec_id = t2.id);

--==============================
select count(*) from public.engine_trackedbox;
	
with -- attach box attributes to boxes
	t1 as 
		(select box_id, text as frame_attr, value as frame_attr_value from -- get attributes for individual boxes
			(select value, box_id, spec_id from public.engine_trackedboxattributeval) as t1
				left join
			(select id, "text" from public.engine_attributespec) as t2
				on
			(t1.spec_id = t2.id))
	select xtl, ytl, xbr, ybr, occluded, frame, outside, track_id, frame_attr, frame_attr_value from
	-- select count(*) from  --  572
		public.engine_trackedbox as t2
			left join
		t1 
			on
		(t2.id = t1.box_id);
		
--=======================================================================
select count(*) from public.engine_objectpathattributeval;

select 
select track_id, text, value from  -- get track attributes
	(select value, spec_id, track_id from public.engine_objectpathattributeval) as t1
		left join
	(select id, "text" from public.engine_attributespec) as t2
		on
	(t1.spec_id = t2.id);
	
--==============================
select * from public.engine_objectpath;
select * from public.engine_label;

with  -- attach path attributes to paths
	t1 as 
		(select track_id, text as attr, value as attr_value from -- get track attributes
			(select value, spec_id, track_id from public.engine_objectpathattributeval) as t1
				left join
			(select id, "text" from public.engine_attributespec) as t2
				on
			(t1.spec_id = t2.id))
	select id, job_id, track_id, attr, attr_value, label from
	--select count(*) from  -- 137
		(select * from 
			t1 
				left join
			public.engine_objectpath as t2
				on
			(t2.id = t1.track_id)) as t1
			left join
		(select id as label_id, name as "label" from public.engine_label) as t2
			using(label_id);
	
		
	
--=======================================================================

with  -- attach boxes to paths
	t1 as 
		(with  -- attach path attributes to paths
			t1 as 
				(select track_id, text as attr, value as attr_value from -- get track attributes
					(select value, spec_id, track_id from public.engine_objectpathattributeval) as t1
						left join
					(select id, "text" from public.engine_attributespec) as t2
						on
					(t1.spec_id = t2.id))
			select id, job_id, track_id, attr, attr_value, label from
				(select * from 
					t1 
						left join
					public.engine_objectpath as t2
						on
					(t2.id = t1.track_id)) as t1
					left join
				(select id as label_id, name as "label" from public.engine_label) as t2
					using(label_id)),
	t2 as 
		(with -- attach box attributes to boxes
			t1 as 
				(select box_id, text as frame_attr, value as frame_attr_value from -- get attributes for individual boxes
					(select value, box_id, spec_id from public.engine_trackedboxattributeval) as t1
						left join
					(select id, "text" from public.engine_attributespec) as t2
						on
					(t1.spec_id = t2.id))
			select xtl, ytl, xbr, ybr, occluded, frame, outside, track_id, frame_attr, frame_attr_value from 
				public.engine_trackedbox as t2
					left join
				t1 
					on
				(t2.id = t1.box_id))
	-- select * from t1;
	select id, job_id, t1.track_id, label, frame, xtl, ytl, xbr, ybr, occluded, outside, attr, attr_value, frame_attr, frame_attr_value from
		t2
			left join
		t1
			on
		(t1.track_id = t2.track_id)
			order by
		job_id, id, frame;
		
-- select * from t2;
	-- select job_id, t1.frame, id, label, xtl, ytl, xbr, ybr, occluded, outside, attr, attr_value, frame_attr, frame_attr_value from
	
--=======================================================================
-- join job to segment
select job_id, task_id from
	(select id as job_id, segment_id from public.engine_job) as t1
		left join
	(select * from public.engine_segment) as t2
		on
	(t1.segment_id = t2.id);
	
-- join job to task
select job_id, name, path from 
	(select job_id, task_id from
		(select id as job_id, segment_id from public.engine_job) as t1
			left join
		(select * from public.engine_segment) as t2
			on
		(t1.segment_id = t2.id)) as t1
		left join
	(select * from public.engine_task) as t2
		on
	(t1.task_id = t2.id);


create table public.cvat_frames as (
with -- attach task to each box
	t1 as
		(with   -- attach boxes to paths
			t1 as 
				(with  -- attach path attributes to paths
					t1 as 
						(select track_id, text as attr, value as attr_value from -- get track attributes
							(select value, spec_id, track_id from public.engine_objectpathattributeval) as t1
								left join
							(select id, "text" from public.engine_attributespec) as t2
								on
							(t1.spec_id = t2.id))
				select id, job_id, track_id, attr, attr_value, label from
					(select * from 
						t1
							join
						public.engine_objectpath as t2
							on
						(t2.id = t1.track_id)) as t1
						left join
					(select id as label_id, name as "label" from public.engine_label) as t2
						using(label_id)),
			t2 as 
				(with -- attach box attributes to boxes
					t1 as 
						(select box_id, text as frame_attr, value as frame_attr_value from  -- get attributes for individual boxes
							(select value, box_id, spec_id from public.engine_trackedboxattributeval) as t1
								left join
							(select id, "text" from public.engine_attributespec) as t2
								on
							(t1.spec_id = t2.id))
					select xtl, ytl, xbr, ybr, occluded, frame, outside, track_id, frame_attr, frame_attr_value from 
						public.engine_trackedbox as t2
							left join
						t1 
							on
						(t2.id = t1.box_id))
			select job_id, t1.track_id, label, frame, xtl, ytl, xbr, ybr, occluded, outside, attr, attr_value, frame_attr, frame_attr_value from
			--select job_id, t1.frame, id, label, xtl, ytl, xbr, ybr, occluded, outside, attr, attr_value, frame_attr, frame_attr_value from
				t1
					left join
				t2
					on
				(t1.id = t2.track_id)
					order by
				job_id, id, frame),
	t2 as (select job_id, name, path from 
				(select job_id, task_id from
					(select id as job_id, segment_id from public.engine_job) as t1
						left join
					(select * from public.engine_segment) as t2
						on
					(t1.segment_id = t2.id)) as t1
					left join
				(select * from public.engine_task) as t2
					on
				(t1.task_id = t2.id))
	select name, path, track_id, label, frame, xtl, ytl, xbr, ybr, occluded, outside, attr, attr_value, frame_attr, frame_attr_value from
		t1
			left join
		t2
			using(job_id));
		
		
			into table public.cvat_frames
			
create table public.cvat_frames;