            drop table dm_temp.old_customer_0524_0;
            create table dm_temp.old_customer_0524_0
            as
               select t1.apply_risk_id, t1.apply_risk_created_at, t1.apply_risk_type, -- {1:单期贷新客, 2:单期贷老客, 3:盲放, 4:多期贷, 5:回流客}
                      t5.apply_risk_id post_rid, t5.apply_risk_apply_id post_aid, t6.apply_oauth_user_id user_id, t6.apply_product_id product_id, 
                      t3.metadata_business_user_info_apply_times apply_times, -- apply_risk_type=5时 {1: 回流新客, >1: 回流老客}
                      t7.individual_identity, t7.individual_name, t7.individual_mobile, 1 period
                 from riskdata.o_apply_risk t1
           inner join riskdata.o_metadata_business_user_info t3
                   on t1.apply_risk_id = t3.metadata_business_user_info_apply_risk_id
            left join riskdata.o_apply_risk_binding t4
                   on t1.apply_risk_id = t4.apply_risk_binding_credit_id
            left join riskdata.o_apply_risk t5
                   on t4.apply_risk_binding_apply_risk_id = t5.apply_risk_id
            left join paydayloan.o_apply t6
                   on t5.apply_risk_apply_id = t6.apply_id
            left join paydayloan.o_oauth_user t8
                   on t6.apply_oauth_user_id = t8.oauth_user_id
            left join paydayloan.o_individual t7
                   on t8.oauth_user_individual_id = t7.individual_id
                where if(t1.apply_risk_created_at<'2018-01-24 16:36:07',t1.apply_risk_source=2, t1.apply_risk_source=19)
                  and t1.apply_risk_status=5
                  and t1.apply_risk_type in (2, 5)
                  and (t1.apply_risk_created_at between '2018-04-18' and '2018-04-24'
                   or t1.apply_risk_created_at between '2018-04-05' and '2018-04-10')
                  and t3.metadata_business_user_info_apply_times > 1;
                
                  
            -- 贷后label
            drop table dm_temp.old_customer_0524_label;
            create table dm_temp.old_customer_0524_label
            as
               select t1.apply_risk_id, t1.period, t2.biz_report_time, t2.biz_report_expect_at, 
                      datediff(if(instr(biz_report_time, '1000-01-01 00:00:00') > 0, from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss'), biz_report_time), biz_report_expect_at) yuqi_day
                 from dm_temp.old_customer_0524_0 t1
            left join riskdata.o_biz_report t2
                   on t1.post_rid = t2.biz_report_apply_risk_id
                  and t2.biz_report_now_period = 1
                  and t2.biz_report_total_period = 1
                  and t2.biz_report_status in (1, 2);
              
            -- 第三方特征
            drop table dm_temp.old_customer_0524_1;
            create table dm_temp.old_customer_0524_1
            as
                  select t1.apply_risk_id, 
                         t3.anti_risk_score ty2, 
                         t4.anti_risk_score ty2forqnn,
                         -- t6.jd_ss_score_payday_sort_score, 
                         t7.baidu_panshi_black_match, t7.baidu_panshi_black_score,
                         t7.baidu_panshi_black_count_level1, t7.baidu_panshi_black_count_level2,
                         t7.baidu_panshi_black_count_level3, 
                         t8.baidu_panshi_duotou_name_match, t8.baidu_panshi_duotou_name_score,
                         t8.baidu_panshi_duotou_name_detail_key, t8.baidu_panshi_duotou_name_detail_val,
                         t8.baidu_panshi_duotou_identity_match, t8.baidu_panshi_duotou_identity_score,
                         t8.baidu_panshi_duotou_identity_detail_key, t8.baidu_panshi_duotou_identity_detail_val,
                         t8.baidu_panshi_duotou_phone_match, t8.baidu_panshi_duotou_phone_score,
                         t8.baidu_panshi_duotou_phone_detail_key, t8.baidu_panshi_duotou_phone_detail_val,
                         t9.baidu_panshi_prea_models, t9.baidu_panshi_prea_score
                    from dm_temp.old_customer_0524_0 t1
               left join riskdata.o_apply_risk_request t2
                      on t1.apply_risk_id = t2.apply_risk_request_apply_risk_id
               left join riskdata.o_anti_fraud t3
                      on t2.apply_risk_request_anti_fraud_version_two_point_zero = t3.anti_risk_request_id
               left join riskdata.o_anti_fraud t4
                      on t2.apply_risk_request_anti_fraud = t4.anti_risk_request_id
               -- left join riskdata.o_apply_risk_data t5
               --       on t1.apply_risk_id = t5. apply_risk_data_apply_risk_id
               --      and t5.apply_risk_data_inside_config_id = 90
               -- left join riskdata.o_jd_ss_score t6
               --       on t5.apply_risk_data_request_id = t6.jd_ss_score_risk_request_id
               left join riskdata.o_baidu_panshi_black t7
                      on t2.apply_risk_request_baidu_panshi_black = t7.baidu_panshi_black_risk_request_id
                     and t7.baidu_panshi_black_match = 1
               left join riskdata.o_baidu_panshi_duotou t8
                      on t2.apply_risk_request_baidu_panshi_duotou = t8.baidu_panshi_duotou_risk_request_id
               left join riskdata.o_baidu_panshi_prea t9
                      on t2.apply_risk_request_baidu_panshi_prea = t9.baidu_panshi_prea_risk_request_id;
                      
            -- metadata数据
            drop table dm_temp.old_customer_0524_2;
            create table dm_temp.old_customer_0524_2
            as
                  select t1.apply_risk_id, 
                         metadata_business_user_info_reg_time reg_time, 
                         metadata_business_user_info_equipment_apps equipment_apps,
                         metadata_business_user_info_equipment_total equipment_total,
                         metadata_business_user_info_last_apply_finish_at last_apply_finish_at,
                         metadata_business_user_info_last_apply_expect_finish_at last_apply_expect_finish_at
                    from dm_temp.old_customer_0524_0 t1
               left join riskdata.o_metadata_business_user_info t2
                      on t1.apply_risk_id=t2.metadata_business_user_info_apply_risk_id;
            
            -- -- 历史借贷记录
            -- drop table dm_temp.old_customer_0524_3;
            -- create table dm_temp.old_customer_0524_3
            -- as
            --    select t1.apply_risk_id, t1.apply_risk_created_at created_at,
            --           t6.apply_risk_id pre_rid, t6.apply_risk_created_at pre_created_at, 
            --           t7.biz_report_expect_at pre_expect_at, t7.biz_report_time pre_finish_at,
            --           biz_report_total_period pre_total_period, biz_report_now_period pre_now_period
            --      from dm_temp.old_customer_0524_0 t1
            -- left join paydayloan.o_apply t4
            --        on t1.post_aid = t4.apply_id
            -- left join paydayloan.o_apply t5
            --        on t4.apply_oauth_user_id = t5.apply_oauth_user_id
            -- left join riskdata.o_apply_risk t6
            --        on t5.apply_id = t6.apply_risk_apply_id
            --       and t6.apply_risk_source = 2
            -- left join riskdata.o_biz_report t7
            --        on t6.apply_risk_id = t7.biz_report_apply_risk_id
            --       and t7.biz_report_status = 1;