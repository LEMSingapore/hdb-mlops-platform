PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL,
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
INSERT INTO alembic_version VALUES('da6fb0208061');
CREATE TABLE experiment_tags (
	"key" VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	experiment_id INTEGER NOT NULL,
	CONSTRAINT experiment_tag_pk PRIMARY KEY ("key", experiment_id),
	FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
);
CREATE TABLE IF NOT EXISTS "runs" (
	run_uuid VARCHAR(32) NOT NULL,
	name VARCHAR(250),
	source_type VARCHAR(20),
	source_name VARCHAR(500),
	entry_point_name VARCHAR(50),
	user_id VARCHAR(256),
	status VARCHAR(9),
	start_time BIGINT,
	end_time BIGINT,
	source_version VARCHAR(50),
	lifecycle_stage VARCHAR(20),
	artifact_uri VARCHAR(200),
	experiment_id INTEGER, deleted_time BIGINT,
	CONSTRAINT run_pk PRIMARY KEY (run_uuid),
	CONSTRAINT source_type CHECK (source_type IN ('NOTEBOOK', 'JOB', 'LOCAL', 'UNKNOWN', 'PROJECT')),
	CONSTRAINT runs_lifecycle_stage CHECK (lifecycle_stage IN ('active', 'deleted')),
	FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CHECK (status IN ('SCHEDULED', 'FAILED', 'FINISHED', 'RUNNING', 'KILLED'))
);
INSERT INTO runs VALUES('b90b186495804fb69cb246143d047dcb','gbr-n1000-lr0.1-d6','UNKNOWN','','','cheeyoungchang','FINISHED',1777345888830,1777346482506,'','active','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/b90b186495804fb69cb246143d047dcb/artifacts',1,NULL);
INSERT INTO runs VALUES('daaef3393bc64e6aacc666b46fa35faa','gbr-n20-lr0.1-d6','UNKNOWN','','','cheeyoungchang','FINISHED',1777440881067,1777440901752,'','active','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/daaef3393bc64e6aacc666b46fa35faa/artifacts',1,NULL);
INSERT INTO runs VALUES('2ec9ef9f05f84e5fa7a31c4c7138f34a','gbr-n1000-lr0.1-d6','UNKNOWN','','','cheeyoungchang','FINISHED',1777441639359,1777441987407,'','active','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/2ec9ef9f05f84e5fa7a31c4c7138f34a/artifacts',1,NULL);
INSERT INTO runs VALUES('d917d0feffd3470894512d9f759e755d','gbr-n1000-lr0.1-d6','UNKNOWN','','','cheeyoungchang','FINISHED',1777443888128,1777444220129,'','active','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/d917d0feffd3470894512d9f759e755d/artifacts',1,NULL);
INSERT INTO runs VALUES('8d7a8bcf23234dbfb273ffd536bdcdb6','gbr-n1000-lr0.1-d6','UNKNOWN','','','cheeyoungchang','FAILED',1777444468258,1777444572955,'','active','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/8d7a8bcf23234dbfb273ffd536bdcdb6/artifacts',1,NULL);
INSERT INTO runs VALUES('a08f4fd052694ab79cd28b5111ba7fb9','gbr-n20-lr0.1-d6','UNKNOWN','','','cheeyoungchang','FINISHED',1777949282621,1777949303341,'','active','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/a08f4fd052694ab79cd28b5111ba7fb9/artifacts',1,NULL);
INSERT INTO runs VALUES('408da2e13b794af59086d35fca34e030','gbr-n1000-lr0.1-d6','UNKNOWN','','','cheeyoungchang','FINISHED',1777951111685,1777951561968,'','active','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/408da2e13b794af59086d35fca34e030/artifacts',1,NULL);
INSERT INTO runs VALUES('26b45ff54a734389ae23eb2ea8b7ab68','gbr-n1000-lr0.1-d6','UNKNOWN','','','cheeyoungchang','FINISHED',1781141255824,1781141597605,'','active','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/26b45ff54a734389ae23eb2ea8b7ab68/artifacts',1,NULL);
CREATE TABLE IF NOT EXISTS "latest_metrics" (
	"key" VARCHAR(250) NOT NULL,
	value FLOAT NOT NULL,
	timestamp BIGINT,
	step BIGINT NOT NULL,
	is_nan BOOLEAN NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT latest_metric_pk PRIMARY KEY ("key", run_uuid),
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid),
	CHECK (is_nan IN (0, 1))
);
INSERT INTO latest_metrics VALUES('train_rmse',27436.07463747865767,1777346452838,0,0,'b90b186495804fb69cb246143d047dcb');
INSERT INTO latest_metrics VALUES('train_mae',19000.79006839992143,1777346452838,0,0,'b90b186495804fb69cb246143d047dcb');
INSERT INTO latest_metrics VALUES('train_r2',0.978254402739351602,1777346452838,0,0,'b90b186495804fb69cb246143d047dcb');
INSERT INTO latest_metrics VALUES('test_rmse',27689.892404665723,1777346452861,0,0,'b90b186495804fb69cb246143d047dcb');
INSERT INTO latest_metrics VALUES('test_mae',19124.6151761772926,1777346452861,0,0,'b90b186495804fb69cb246143d047dcb');
INSERT INTO latest_metrics VALUES('test_r2',0.977848665601068978,1777346452861,0,0,'b90b186495804fb69cb246143d047dcb');
INSERT INTO latest_metrics VALUES('train_rmse',96854.3083822345652,1777440889931,0,0,'daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO latest_metrics VALUES('train_mae',69668.63334265777667,1777440889931,0,0,'daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO latest_metrics VALUES('train_r2',0.7290025042477925598,1777440889931,0,0,'daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO latest_metrics VALUES('test_rmse',97050.3141313651867,1777440889956,0,0,'daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO latest_metrics VALUES('test_mae',69684.31503631916713,1777440889956,0,0,'daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO latest_metrics VALUES('test_r2',0.7278857735542606511,1777440889956,0,0,'daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO latest_metrics VALUES('train_rmse',28751.94725278792976,1777441958638,0,0,'2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO latest_metrics VALUES('train_mae',19722.5622199285499,1777441958638,0,0,'2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO latest_metrics VALUES('train_r2',0.976118482746137994,1777441958638,0,0,'2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO latest_metrics VALUES('test_rmse',29296.14859625095051,1777441958663,0,0,'2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO latest_metrics VALUES('test_mae',20007.79462039327701,1777441958663,0,0,'2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO latest_metrics VALUES('test_r2',0.975204183341829788,1777441958663,0,0,'2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO latest_metrics VALUES('train_rmse',28751.94725278792976,1777444192018,0,0,'d917d0feffd3470894512d9f759e755d');
INSERT INTO latest_metrics VALUES('train_mae',19722.5622199285499,1777444192018,0,0,'d917d0feffd3470894512d9f759e755d');
INSERT INTO latest_metrics VALUES('train_r2',0.976118482746137994,1777444192018,0,0,'d917d0feffd3470894512d9f759e755d');
INSERT INTO latest_metrics VALUES('test_rmse',29296.14859625095051,1777444192041,0,0,'d917d0feffd3470894512d9f759e755d');
INSERT INTO latest_metrics VALUES('test_mae',20007.79462039327701,1777444192041,0,0,'d917d0feffd3470894512d9f759e755d');
INSERT INTO latest_metrics VALUES('test_r2',0.975204183341829788,1777444192041,0,0,'d917d0feffd3470894512d9f759e755d');
INSERT INTO latest_metrics VALUES('train_rmse',96854.3083822345652,1777949292629,0,0,'a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO latest_metrics VALUES('train_mae',69668.63334265777667,1777949292629,0,0,'a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO latest_metrics VALUES('train_r2',0.7290025042477925598,1777949292629,0,0,'a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO latest_metrics VALUES('test_rmse',97050.3141313651867,1777949292644,0,0,'a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO latest_metrics VALUES('test_mae',69684.31503631916713,1777949292644,0,0,'a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO latest_metrics VALUES('test_r2',0.7278857735542606511,1777949292644,0,0,'a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO latest_metrics VALUES('train_rmse',28751.94725278792976,1777951534594,0,0,'408da2e13b794af59086d35fca34e030');
INSERT INTO latest_metrics VALUES('train_mae',19722.5622199285499,1777951534594,0,0,'408da2e13b794af59086d35fca34e030');
INSERT INTO latest_metrics VALUES('train_r2',0.976118482746137994,1777951534594,0,0,'408da2e13b794af59086d35fca34e030');
INSERT INTO latest_metrics VALUES('test_rmse',29296.14859625095051,1777951534616,0,0,'408da2e13b794af59086d35fca34e030');
INSERT INTO latest_metrics VALUES('test_mae',20007.79462039327701,1777951534616,0,0,'408da2e13b794af59086d35fca34e030');
INSERT INTO latest_metrics VALUES('test_r2',0.975204183341829788,1777951534616,0,0,'408da2e13b794af59086d35fca34e030');
INSERT INTO latest_metrics VALUES('train_rmse',28736.06471350958964,1781141569559,0,0,'26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO latest_metrics VALUES('train_mae',19719.68879849981021,1781141569559,0,0,'26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO latest_metrics VALUES('train_r2',0.976144859698648171,1781141569559,0,0,'26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO latest_metrics VALUES('test_rmse',29270.82710165078606,1781141569597,0,0,'26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO latest_metrics VALUES('test_mae',19999.30753476619065,1781141569597,0,0,'26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO latest_metrics VALUES('test_r2',0.975247028277152261,1781141569597,0,0,'26b45ff54a734389ae23eb2ea8b7ab68');
CREATE TABLE IF NOT EXISTS "metrics" (
	"key" VARCHAR(250) NOT NULL,
	value FLOAT NOT NULL,
	timestamp BIGINT NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	step BIGINT DEFAULT '0' NOT NULL,
	is_nan BOOLEAN DEFAULT '0' NOT NULL,
	CONSTRAINT metric_pk PRIMARY KEY ("key", timestamp, step, run_uuid, value, is_nan),
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid),
	CHECK (is_nan IN (0, 1))
);
INSERT INTO metrics VALUES('train_rmse',27436.07463747865767,1777346452838,'b90b186495804fb69cb246143d047dcb',0,0);
INSERT INTO metrics VALUES('train_mae',19000.79006839992143,1777346452838,'b90b186495804fb69cb246143d047dcb',0,0);
INSERT INTO metrics VALUES('train_r2',0.978254402739351602,1777346452838,'b90b186495804fb69cb246143d047dcb',0,0);
INSERT INTO metrics VALUES('test_rmse',27689.892404665723,1777346452861,'b90b186495804fb69cb246143d047dcb',0,0);
INSERT INTO metrics VALUES('test_mae',19124.6151761772926,1777346452861,'b90b186495804fb69cb246143d047dcb',0,0);
INSERT INTO metrics VALUES('test_r2',0.977848665601068978,1777346452861,'b90b186495804fb69cb246143d047dcb',0,0);
INSERT INTO metrics VALUES('train_rmse',96854.3083822345652,1777440889931,'daaef3393bc64e6aacc666b46fa35faa',0,0);
INSERT INTO metrics VALUES('train_mae',69668.63334265777667,1777440889931,'daaef3393bc64e6aacc666b46fa35faa',0,0);
INSERT INTO metrics VALUES('train_r2',0.7290025042477925598,1777440889931,'daaef3393bc64e6aacc666b46fa35faa',0,0);
INSERT INTO metrics VALUES('test_rmse',97050.3141313651867,1777440889956,'daaef3393bc64e6aacc666b46fa35faa',0,0);
INSERT INTO metrics VALUES('test_mae',69684.31503631916713,1777440889956,'daaef3393bc64e6aacc666b46fa35faa',0,0);
INSERT INTO metrics VALUES('test_r2',0.7278857735542606511,1777440889956,'daaef3393bc64e6aacc666b46fa35faa',0,0);
INSERT INTO metrics VALUES('train_rmse',28751.94725278792976,1777441958638,'2ec9ef9f05f84e5fa7a31c4c7138f34a',0,0);
INSERT INTO metrics VALUES('train_mae',19722.5622199285499,1777441958638,'2ec9ef9f05f84e5fa7a31c4c7138f34a',0,0);
INSERT INTO metrics VALUES('train_r2',0.976118482746137994,1777441958638,'2ec9ef9f05f84e5fa7a31c4c7138f34a',0,0);
INSERT INTO metrics VALUES('test_rmse',29296.14859625095051,1777441958663,'2ec9ef9f05f84e5fa7a31c4c7138f34a',0,0);
INSERT INTO metrics VALUES('test_mae',20007.79462039327701,1777441958663,'2ec9ef9f05f84e5fa7a31c4c7138f34a',0,0);
INSERT INTO metrics VALUES('test_r2',0.975204183341829788,1777441958663,'2ec9ef9f05f84e5fa7a31c4c7138f34a',0,0);
INSERT INTO metrics VALUES('train_rmse',28751.94725278792976,1777444192018,'d917d0feffd3470894512d9f759e755d',0,0);
INSERT INTO metrics VALUES('train_mae',19722.5622199285499,1777444192018,'d917d0feffd3470894512d9f759e755d',0,0);
INSERT INTO metrics VALUES('train_r2',0.976118482746137994,1777444192018,'d917d0feffd3470894512d9f759e755d',0,0);
INSERT INTO metrics VALUES('test_rmse',29296.14859625095051,1777444192041,'d917d0feffd3470894512d9f759e755d',0,0);
INSERT INTO metrics VALUES('test_mae',20007.79462039327701,1777444192041,'d917d0feffd3470894512d9f759e755d',0,0);
INSERT INTO metrics VALUES('test_r2',0.975204183341829788,1777444192041,'d917d0feffd3470894512d9f759e755d',0,0);
INSERT INTO metrics VALUES('train_rmse',96854.3083822345652,1777949292629,'a08f4fd052694ab79cd28b5111ba7fb9',0,0);
INSERT INTO metrics VALUES('train_mae',69668.63334265777667,1777949292629,'a08f4fd052694ab79cd28b5111ba7fb9',0,0);
INSERT INTO metrics VALUES('train_r2',0.7290025042477925598,1777949292629,'a08f4fd052694ab79cd28b5111ba7fb9',0,0);
INSERT INTO metrics VALUES('test_rmse',97050.3141313651867,1777949292644,'a08f4fd052694ab79cd28b5111ba7fb9',0,0);
INSERT INTO metrics VALUES('test_mae',69684.31503631916713,1777949292644,'a08f4fd052694ab79cd28b5111ba7fb9',0,0);
INSERT INTO metrics VALUES('test_r2',0.7278857735542606511,1777949292644,'a08f4fd052694ab79cd28b5111ba7fb9',0,0);
INSERT INTO metrics VALUES('train_rmse',28751.94725278792976,1777951534594,'408da2e13b794af59086d35fca34e030',0,0);
INSERT INTO metrics VALUES('train_mae',19722.5622199285499,1777951534594,'408da2e13b794af59086d35fca34e030',0,0);
INSERT INTO metrics VALUES('train_r2',0.976118482746137994,1777951534594,'408da2e13b794af59086d35fca34e030',0,0);
INSERT INTO metrics VALUES('test_rmse',29296.14859625095051,1777951534616,'408da2e13b794af59086d35fca34e030',0,0);
INSERT INTO metrics VALUES('test_mae',20007.79462039327701,1777951534616,'408da2e13b794af59086d35fca34e030',0,0);
INSERT INTO metrics VALUES('test_r2',0.975204183341829788,1777951534616,'408da2e13b794af59086d35fca34e030',0,0);
INSERT INTO metrics VALUES('train_rmse',28736.06471350958964,1781141569559,'26b45ff54a734389ae23eb2ea8b7ab68',0,0);
INSERT INTO metrics VALUES('train_mae',19719.68879849981021,1781141569559,'26b45ff54a734389ae23eb2ea8b7ab68',0,0);
INSERT INTO metrics VALUES('train_r2',0.976144859698648171,1781141569559,'26b45ff54a734389ae23eb2ea8b7ab68',0,0);
INSERT INTO metrics VALUES('test_rmse',29270.82710165078606,1781141569597,'26b45ff54a734389ae23eb2ea8b7ab68',0,0);
INSERT INTO metrics VALUES('test_mae',19999.30753476619065,1781141569597,'26b45ff54a734389ae23eb2ea8b7ab68',0,0);
INSERT INTO metrics VALUES('test_r2',0.975247028277152261,1781141569597,'26b45ff54a734389ae23eb2ea8b7ab68',0,0);
CREATE TABLE inputs (
	input_uuid VARCHAR(36) NOT NULL,
	source_type VARCHAR(36) NOT NULL,
	source_id VARCHAR(36) NOT NULL,
	destination_type VARCHAR(36) NOT NULL,
	destination_id VARCHAR(36) NOT NULL, step BIGINT DEFAULT '0' NOT NULL,
	CONSTRAINT inputs_pk PRIMARY KEY (source_type, source_id, destination_type, destination_id)
);
INSERT INTO inputs VALUES('3f3a8c65a92d4d2e85a4d1d0928b878c','RUN_OUTPUT','b90b186495804fb69cb246143d047dcb','MODEL_OUTPUT','m-381dfab095e04fd0985077c559c994e7',0);
INSERT INTO inputs VALUES('06677b971495499081bcc6f9f395c6da','RUN_OUTPUT','daaef3393bc64e6aacc666b46fa35faa','MODEL_OUTPUT','m-f87f54e6757a48489e0f106fd862388c',0);
INSERT INTO inputs VALUES('6315a8cad5974c8a9e421619a4550378','RUN_OUTPUT','2ec9ef9f05f84e5fa7a31c4c7138f34a','MODEL_OUTPUT','m-a2f7dc39cffa44f9b114328acfefdc0f',0);
INSERT INTO inputs VALUES('55b6659b9a234e13889ff0298070948a','RUN_OUTPUT','d917d0feffd3470894512d9f759e755d','MODEL_OUTPUT','m-bf10a6cce5bd49549963d2db305cbcfd',0);
INSERT INTO inputs VALUES('f920c491e5554e388d1b28366d8231be','RUN_OUTPUT','a08f4fd052694ab79cd28b5111ba7fb9','MODEL_OUTPUT','m-82a509e18d914655aa3e3e356e8ab7a5',0);
INSERT INTO inputs VALUES('37fefff28ff84a5abef364641acfa145','RUN_OUTPUT','408da2e13b794af59086d35fca34e030','MODEL_OUTPUT','m-cb9ee5b748924c029ee932c3f7c078ba',0);
INSERT INTO inputs VALUES('ce2fa519ad78423c98075af939640f94','RUN_OUTPUT','26b45ff54a734389ae23eb2ea8b7ab68','MODEL_OUTPUT','m-72a82cd491384c26811e7927af1ca6fd',0);
CREATE TABLE input_tags (
	input_uuid VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	value VARCHAR(500) NOT NULL,
	CONSTRAINT input_tags_pk PRIMARY KEY (input_uuid, name)
);
CREATE TABLE IF NOT EXISTS "params" (
	"key" VARCHAR(250) NOT NULL,
	value VARCHAR(8000) NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT param_pk PRIMARY KEY ("key", run_uuid),
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
);
INSERT INTO params VALUES('n_estimators','1000','b90b186495804fb69cb246143d047dcb');
INSERT INTO params VALUES('learning_rate','0.1','b90b186495804fb69cb246143d047dcb');
INSERT INTO params VALUES('max_depth','6','b90b186495804fb69cb246143d047dcb');
INSERT INTO params VALUES('min_samples_leaf','9','b90b186495804fb69cb246143d047dcb');
INSERT INTO params VALUES('max_features','0.1','b90b186495804fb69cb246143d047dcb');
INSERT INTO params VALUES('test_size','0.2','b90b186495804fb69cb246143d047dcb');
INSERT INTO params VALUES('random_state','7','b90b186495804fb69cb246143d047dcb');
INSERT INTO params VALUES('n_estimators','20','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO params VALUES('learning_rate','0.1','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO params VALUES('max_depth','6','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO params VALUES('min_samples_leaf','9','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO params VALUES('max_features','0.1','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO params VALUES('test_size','0.2','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO params VALUES('random_state','7','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO params VALUES('n_estimators','1000','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO params VALUES('learning_rate','0.1','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO params VALUES('max_depth','6','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO params VALUES('min_samples_leaf','9','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO params VALUES('max_features','0.1','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO params VALUES('test_size','0.2','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO params VALUES('random_state','7','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO params VALUES('n_estimators','1000','d917d0feffd3470894512d9f759e755d');
INSERT INTO params VALUES('learning_rate','0.1','d917d0feffd3470894512d9f759e755d');
INSERT INTO params VALUES('max_depth','6','d917d0feffd3470894512d9f759e755d');
INSERT INTO params VALUES('min_samples_leaf','9','d917d0feffd3470894512d9f759e755d');
INSERT INTO params VALUES('max_features','0.1','d917d0feffd3470894512d9f759e755d');
INSERT INTO params VALUES('test_size','0.2','d917d0feffd3470894512d9f759e755d');
INSERT INTO params VALUES('random_state','7','d917d0feffd3470894512d9f759e755d');
INSERT INTO params VALUES('n_estimators','1000','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO params VALUES('learning_rate','0.1','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO params VALUES('max_depth','6','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO params VALUES('min_samples_leaf','9','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO params VALUES('max_features','0.1','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO params VALUES('test_size','0.2','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO params VALUES('random_state','7','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO params VALUES('n_estimators','20','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO params VALUES('learning_rate','0.1','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO params VALUES('max_depth','6','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO params VALUES('min_samples_leaf','9','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO params VALUES('max_features','0.1','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO params VALUES('test_size','0.2','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO params VALUES('random_state','7','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO params VALUES('n_estimators','1000','408da2e13b794af59086d35fca34e030');
INSERT INTO params VALUES('learning_rate','0.1','408da2e13b794af59086d35fca34e030');
INSERT INTO params VALUES('max_depth','6','408da2e13b794af59086d35fca34e030');
INSERT INTO params VALUES('min_samples_leaf','9','408da2e13b794af59086d35fca34e030');
INSERT INTO params VALUES('max_features','0.1','408da2e13b794af59086d35fca34e030');
INSERT INTO params VALUES('test_size','0.2','408da2e13b794af59086d35fca34e030');
INSERT INTO params VALUES('random_state','7','408da2e13b794af59086d35fca34e030');
INSERT INTO params VALUES('n_estimators','1000','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO params VALUES('learning_rate','0.1','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO params VALUES('max_depth','6','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO params VALUES('min_samples_leaf','9','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO params VALUES('max_features','0.1','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO params VALUES('test_size','0.2','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO params VALUES('random_state','7','26b45ff54a734389ae23eb2ea8b7ab68');
CREATE TABLE trace_info (
	request_id VARCHAR(50) NOT NULL,
	experiment_id INTEGER NOT NULL,
	timestamp_ms BIGINT NOT NULL,
	execution_time_ms BIGINT,
	status VARCHAR(50) NOT NULL, client_request_id VARCHAR(50), request_preview VARCHAR(1000), response_preview VARCHAR(1000), db_payload_generation INTEGER DEFAULT '0' NOT NULL,
	CONSTRAINT trace_info_pk PRIMARY KEY (request_id),
	CONSTRAINT fk_trace_info_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
);
CREATE TABLE IF NOT EXISTS "trace_tags" (
	"key" VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	request_id VARCHAR(50) NOT NULL,
	CONSTRAINT trace_tag_pk PRIMARY KEY ("key", request_id),
	CONSTRAINT fk_trace_tags_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "trace_request_metadata" (
	"key" VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	request_id VARCHAR(50) NOT NULL,
	CONSTRAINT trace_request_metadata_pk PRIMARY KEY ("key", request_id),
	CONSTRAINT fk_trace_request_metadata_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "tags" (
	"key" VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT tag_pk PRIMARY KEY ("key", run_uuid),
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
);
INSERT INTO tags VALUES('mlflow.user','cheeyoungchang','b90b186495804fb69cb246143d047dcb');
INSERT INTO tags VALUES('mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py','b90b186495804fb69cb246143d047dcb');
INSERT INTO tags VALUES('mlflow.source.type','LOCAL','b90b186495804fb69cb246143d047dcb');
INSERT INTO tags VALUES('mlflow.source.git.commit','7c7096693079143753d2cfb938e9be23aa459873','b90b186495804fb69cb246143d047dcb');
INSERT INTO tags VALUES('mlflow.runName','gbr-n1000-lr0.1-d6','b90b186495804fb69cb246143d047dcb');
INSERT INTO tags VALUES('mlflow.user','cheeyoungchang','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO tags VALUES('mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO tags VALUES('mlflow.source.type','LOCAL','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO tags VALUES('mlflow.source.git.commit','6e0bb5dbe913c22f91de7e3e0b4ffe32c6a7e425','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO tags VALUES('mlflow.runName','gbr-n20-lr0.1-d6','daaef3393bc64e6aacc666b46fa35faa');
INSERT INTO tags VALUES('mlflow.user','cheeyoungchang','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO tags VALUES('mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO tags VALUES('mlflow.source.type','LOCAL','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO tags VALUES('mlflow.source.git.commit','6e0bb5dbe913c22f91de7e3e0b4ffe32c6a7e425','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO tags VALUES('mlflow.runName','gbr-n1000-lr0.1-d6','2ec9ef9f05f84e5fa7a31c4c7138f34a');
INSERT INTO tags VALUES('mlflow.user','cheeyoungchang','d917d0feffd3470894512d9f759e755d');
INSERT INTO tags VALUES('mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py','d917d0feffd3470894512d9f759e755d');
INSERT INTO tags VALUES('mlflow.source.type','LOCAL','d917d0feffd3470894512d9f759e755d');
INSERT INTO tags VALUES('mlflow.source.git.commit','6e0bb5dbe913c22f91de7e3e0b4ffe32c6a7e425','d917d0feffd3470894512d9f759e755d');
INSERT INTO tags VALUES('mlflow.runName','gbr-n1000-lr0.1-d6','d917d0feffd3470894512d9f759e755d');
INSERT INTO tags VALUES('mlflow.user','cheeyoungchang','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO tags VALUES('mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO tags VALUES('mlflow.source.type','LOCAL','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO tags VALUES('mlflow.source.git.commit','6e0bb5dbe913c22f91de7e3e0b4ffe32c6a7e425','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO tags VALUES('mlflow.runName','gbr-n1000-lr0.1-d6','8d7a8bcf23234dbfb273ffd536bdcdb6');
INSERT INTO tags VALUES('mlflow.user','cheeyoungchang','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO tags VALUES('mlflow.source.name','/Users/cheeyoungchang/Projects/hdb-mlops-platform/src/training/train.py','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO tags VALUES('mlflow.source.type','LOCAL','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO tags VALUES('mlflow.source.git.commit','e21c833e6c790599418340ad536bdb451ac0935f','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO tags VALUES('mlflow.runName','gbr-n20-lr0.1-d6','a08f4fd052694ab79cd28b5111ba7fb9');
INSERT INTO tags VALUES('mlflow.user','cheeyoungchang','408da2e13b794af59086d35fca34e030');
INSERT INTO tags VALUES('mlflow.source.name','/Users/cheeyoungchang/Projects/hdb-mlops-platform/src/training/train.py','408da2e13b794af59086d35fca34e030');
INSERT INTO tags VALUES('mlflow.source.type','LOCAL','408da2e13b794af59086d35fca34e030');
INSERT INTO tags VALUES('mlflow.source.git.commit','e21c833e6c790599418340ad536bdb451ac0935f','408da2e13b794af59086d35fca34e030');
INSERT INTO tags VALUES('mlflow.runName','gbr-n1000-lr0.1-d6','408da2e13b794af59086d35fca34e030');
INSERT INTO tags VALUES('mlflow.user','cheeyoungchang','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO tags VALUES('mlflow.source.name','/Users/cheeyoungchang/Projects/hdb-mlops-platform/src/training/train.py','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO tags VALUES('mlflow.source.type','LOCAL','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO tags VALUES('mlflow.source.git.commit','382cd68db0974ef9674dfd0ea818cc07ad5715d1','26b45ff54a734389ae23eb2ea8b7ab68');
INSERT INTO tags VALUES('mlflow.runName','gbr-n1000-lr0.1-d6','26b45ff54a734389ae23eb2ea8b7ab68');
CREATE TABLE IF NOT EXISTS "datasets" (
	dataset_uuid VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(500) NOT NULL,
	digest VARCHAR(36) NOT NULL,
	dataset_source_type VARCHAR(36) NOT NULL,
	dataset_source TEXT NOT NULL,
	dataset_schema TEXT,
	dataset_profile TEXT,
	CONSTRAINT dataset_pk PRIMARY KEY (experiment_id, name, digest),
	CONSTRAINT fk_datasets_experiment_id_experiments FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE
);
CREATE TABLE logged_models (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(500) NOT NULL,
	artifact_location VARCHAR(1000) NOT NULL,
	creation_timestamp_ms BIGINT NOT NULL,
	last_updated_timestamp_ms BIGINT NOT NULL,
	status INTEGER NOT NULL,
	lifecycle_stage VARCHAR(32),
	model_type VARCHAR(500),
	source_run_id VARCHAR(32),
	status_message VARCHAR(1000),
	CONSTRAINT logged_models_pk PRIMARY KEY (model_id),
	CONSTRAINT logged_models_lifecycle_stage_check CHECK (lifecycle_stage IN ('active', 'deleted')),
	CONSTRAINT fk_logged_models_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE
);
INSERT INTO logged_models VALUES('m-381dfab095e04fd0985077c559c994e7',1,'model','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-381dfab095e04fd0985077c559c994e7/artifacts',1777346474108,1777346482450,2,'active',NULL,'b90b186495804fb69cb246143d047dcb',NULL);
INSERT INTO logged_models VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'model','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-f87f54e6757a48489e0f106fd862388c/artifacts',1777440897401,1777440901700,2,'active',NULL,'daaef3393bc64e6aacc666b46fa35faa',NULL);
INSERT INTO logged_models VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'model','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-a2f7dc39cffa44f9b114328acfefdc0f/artifacts',1777441982270,1777441987355,2,'active',NULL,'2ec9ef9f05f84e5fa7a31c4c7138f34a',NULL);
INSERT INTO logged_models VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'model','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-bf10a6cce5bd49549963d2db305cbcfd/artifacts',1777444215202,1777444220077,2,'active',NULL,'d917d0feffd3470894512d9f759e755d',NULL);
INSERT INTO logged_models VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'model','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-82a509e18d914655aa3e3e356e8ab7a5/artifacts',1777949299090,1777949303290,2,'active',NULL,'a08f4fd052694ab79cd28b5111ba7fb9',NULL);
INSERT INTO logged_models VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'model','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-cb9ee5b748924c029ee932c3f7c078ba/artifacts',1777951556731,1777951561904,2,'active',NULL,'408da2e13b794af59086d35fca34e030',NULL);
INSERT INTO logged_models VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'model','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-72a82cd491384c26811e7927af1ca6fd/artifacts',1781141592697,1781141597546,2,'active',NULL,'26b45ff54a734389ae23eb2ea8b7ab68',NULL);
CREATE TABLE logged_model_metrics (
	model_id VARCHAR(36) NOT NULL,
	metric_name VARCHAR(500) NOT NULL,
	metric_timestamp_ms BIGINT NOT NULL,
	metric_step BIGINT NOT NULL,
	metric_value FLOAT,
	experiment_id INTEGER NOT NULL,
	run_id VARCHAR(32) NOT NULL,
	dataset_uuid VARCHAR(36),
	dataset_name VARCHAR(500),
	dataset_digest VARCHAR(36),
	CONSTRAINT logged_model_metrics_pk PRIMARY KEY (model_id, metric_name, metric_timestamp_ms, metric_step, run_id),
	CONSTRAINT fk_logged_model_metrics_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_metrics_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE,
	CONSTRAINT fk_logged_model_metrics_run_id FOREIGN KEY(run_id) REFERENCES runs (run_uuid) ON DELETE CASCADE
);
INSERT INTO logged_model_metrics VALUES('m-381dfab095e04fd0985077c559c994e7','train_rmse',1777346452838,0,27436.07463747865767,1,'b90b186495804fb69cb246143d047dcb',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-381dfab095e04fd0985077c559c994e7','train_mae',1777346452838,0,19000.79006839992143,1,'b90b186495804fb69cb246143d047dcb',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-381dfab095e04fd0985077c559c994e7','train_r2',1777346452838,0,0.978254402739351602,1,'b90b186495804fb69cb246143d047dcb',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-381dfab095e04fd0985077c559c994e7','test_rmse',1777346452861,0,27689.892404665723,1,'b90b186495804fb69cb246143d047dcb',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-381dfab095e04fd0985077c559c994e7','test_mae',1777346452861,0,19124.6151761772926,1,'b90b186495804fb69cb246143d047dcb',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-381dfab095e04fd0985077c559c994e7','test_r2',1777346452861,0,0.977848665601068978,1,'b90b186495804fb69cb246143d047dcb',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-f87f54e6757a48489e0f106fd862388c','train_rmse',1777440889931,0,96854.3083822345652,1,'daaef3393bc64e6aacc666b46fa35faa',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-f87f54e6757a48489e0f106fd862388c','train_mae',1777440889931,0,69668.63334265777667,1,'daaef3393bc64e6aacc666b46fa35faa',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-f87f54e6757a48489e0f106fd862388c','train_r2',1777440889931,0,0.7290025042477925598,1,'daaef3393bc64e6aacc666b46fa35faa',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-f87f54e6757a48489e0f106fd862388c','test_rmse',1777440889956,0,97050.3141313651867,1,'daaef3393bc64e6aacc666b46fa35faa',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-f87f54e6757a48489e0f106fd862388c','test_mae',1777440889956,0,69684.31503631916713,1,'daaef3393bc64e6aacc666b46fa35faa',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-f87f54e6757a48489e0f106fd862388c','test_r2',1777440889956,0,0.7278857735542606511,1,'daaef3393bc64e6aacc666b46fa35faa',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f','train_rmse',1777441958638,0,28751.94725278792976,1,'2ec9ef9f05f84e5fa7a31c4c7138f34a',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f','train_mae',1777441958638,0,19722.5622199285499,1,'2ec9ef9f05f84e5fa7a31c4c7138f34a',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f','train_r2',1777441958638,0,0.976118482746137994,1,'2ec9ef9f05f84e5fa7a31c4c7138f34a',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f','test_rmse',1777441958663,0,29296.14859625095051,1,'2ec9ef9f05f84e5fa7a31c4c7138f34a',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f','test_mae',1777441958663,0,20007.79462039327701,1,'2ec9ef9f05f84e5fa7a31c4c7138f34a',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f','test_r2',1777441958663,0,0.975204183341829788,1,'2ec9ef9f05f84e5fa7a31c4c7138f34a',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-bf10a6cce5bd49549963d2db305cbcfd','train_rmse',1777444192018,0,28751.94725278792976,1,'d917d0feffd3470894512d9f759e755d',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-bf10a6cce5bd49549963d2db305cbcfd','train_mae',1777444192018,0,19722.5622199285499,1,'d917d0feffd3470894512d9f759e755d',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-bf10a6cce5bd49549963d2db305cbcfd','train_r2',1777444192018,0,0.976118482746137994,1,'d917d0feffd3470894512d9f759e755d',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-bf10a6cce5bd49549963d2db305cbcfd','test_rmse',1777444192041,0,29296.14859625095051,1,'d917d0feffd3470894512d9f759e755d',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-bf10a6cce5bd49549963d2db305cbcfd','test_mae',1777444192041,0,20007.79462039327701,1,'d917d0feffd3470894512d9f759e755d',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-bf10a6cce5bd49549963d2db305cbcfd','test_r2',1777444192041,0,0.975204183341829788,1,'d917d0feffd3470894512d9f759e755d',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-82a509e18d914655aa3e3e356e8ab7a5','train_rmse',1777949292629,0,96854.3083822345652,1,'a08f4fd052694ab79cd28b5111ba7fb9',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-82a509e18d914655aa3e3e356e8ab7a5','train_mae',1777949292629,0,69668.63334265777667,1,'a08f4fd052694ab79cd28b5111ba7fb9',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-82a509e18d914655aa3e3e356e8ab7a5','train_r2',1777949292629,0,0.7290025042477925598,1,'a08f4fd052694ab79cd28b5111ba7fb9',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-82a509e18d914655aa3e3e356e8ab7a5','test_rmse',1777949292644,0,97050.3141313651867,1,'a08f4fd052694ab79cd28b5111ba7fb9',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-82a509e18d914655aa3e3e356e8ab7a5','test_mae',1777949292644,0,69684.31503631916713,1,'a08f4fd052694ab79cd28b5111ba7fb9',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-82a509e18d914655aa3e3e356e8ab7a5','test_r2',1777949292644,0,0.7278857735542606511,1,'a08f4fd052694ab79cd28b5111ba7fb9',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-cb9ee5b748924c029ee932c3f7c078ba','train_rmse',1777951534594,0,28751.94725278792976,1,'408da2e13b794af59086d35fca34e030',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-cb9ee5b748924c029ee932c3f7c078ba','train_mae',1777951534594,0,19722.5622199285499,1,'408da2e13b794af59086d35fca34e030',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-cb9ee5b748924c029ee932c3f7c078ba','train_r2',1777951534594,0,0.976118482746137994,1,'408da2e13b794af59086d35fca34e030',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-cb9ee5b748924c029ee932c3f7c078ba','test_rmse',1777951534616,0,29296.14859625095051,1,'408da2e13b794af59086d35fca34e030',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-cb9ee5b748924c029ee932c3f7c078ba','test_mae',1777951534616,0,20007.79462039327701,1,'408da2e13b794af59086d35fca34e030',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-cb9ee5b748924c029ee932c3f7c078ba','test_r2',1777951534616,0,0.975204183341829788,1,'408da2e13b794af59086d35fca34e030',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-72a82cd491384c26811e7927af1ca6fd','train_rmse',1781141569559,0,28736.06471350958964,1,'26b45ff54a734389ae23eb2ea8b7ab68',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-72a82cd491384c26811e7927af1ca6fd','train_mae',1781141569559,0,19719.68879849981021,1,'26b45ff54a734389ae23eb2ea8b7ab68',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-72a82cd491384c26811e7927af1ca6fd','train_r2',1781141569559,0,0.976144859698648171,1,'26b45ff54a734389ae23eb2ea8b7ab68',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-72a82cd491384c26811e7927af1ca6fd','test_rmse',1781141569597,0,29270.82710165078606,1,'26b45ff54a734389ae23eb2ea8b7ab68',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-72a82cd491384c26811e7927af1ca6fd','test_mae',1781141569597,0,19999.30753476619065,1,'26b45ff54a734389ae23eb2ea8b7ab68',NULL,NULL,NULL);
INSERT INTO logged_model_metrics VALUES('m-72a82cd491384c26811e7927af1ca6fd','test_r2',1781141569597,0,0.975247028277152261,1,'26b45ff54a734389ae23eb2ea8b7ab68',NULL,NULL,NULL);
CREATE TABLE logged_model_params (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	param_key VARCHAR(255) NOT NULL,
	param_value TEXT NOT NULL,
	CONSTRAINT logged_model_params_pk PRIMARY KEY (model_id, param_key),
	CONSTRAINT fk_logged_model_params_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_params_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE
);
INSERT INTO logged_model_params VALUES('m-381dfab095e04fd0985077c559c994e7',1,'n_estimators','1000');
INSERT INTO logged_model_params VALUES('m-381dfab095e04fd0985077c559c994e7',1,'learning_rate','0.1');
INSERT INTO logged_model_params VALUES('m-381dfab095e04fd0985077c559c994e7',1,'max_depth','6');
INSERT INTO logged_model_params VALUES('m-381dfab095e04fd0985077c559c994e7',1,'min_samples_leaf','9');
INSERT INTO logged_model_params VALUES('m-381dfab095e04fd0985077c559c994e7',1,'max_features','0.1');
INSERT INTO logged_model_params VALUES('m-381dfab095e04fd0985077c559c994e7',1,'test_size','0.2');
INSERT INTO logged_model_params VALUES('m-381dfab095e04fd0985077c559c994e7',1,'random_state','7');
INSERT INTO logged_model_params VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'n_estimators','20');
INSERT INTO logged_model_params VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'learning_rate','0.1');
INSERT INTO logged_model_params VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'max_depth','6');
INSERT INTO logged_model_params VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'min_samples_leaf','9');
INSERT INTO logged_model_params VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'max_features','0.1');
INSERT INTO logged_model_params VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'test_size','0.2');
INSERT INTO logged_model_params VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'random_state','7');
INSERT INTO logged_model_params VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'n_estimators','1000');
INSERT INTO logged_model_params VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'learning_rate','0.1');
INSERT INTO logged_model_params VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'max_depth','6');
INSERT INTO logged_model_params VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'min_samples_leaf','9');
INSERT INTO logged_model_params VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'max_features','0.1');
INSERT INTO logged_model_params VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'test_size','0.2');
INSERT INTO logged_model_params VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'random_state','7');
INSERT INTO logged_model_params VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'n_estimators','1000');
INSERT INTO logged_model_params VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'learning_rate','0.1');
INSERT INTO logged_model_params VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'max_depth','6');
INSERT INTO logged_model_params VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'min_samples_leaf','9');
INSERT INTO logged_model_params VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'max_features','0.1');
INSERT INTO logged_model_params VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'test_size','0.2');
INSERT INTO logged_model_params VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'random_state','7');
INSERT INTO logged_model_params VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'n_estimators','20');
INSERT INTO logged_model_params VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'learning_rate','0.1');
INSERT INTO logged_model_params VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'max_depth','6');
INSERT INTO logged_model_params VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'min_samples_leaf','9');
INSERT INTO logged_model_params VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'max_features','0.1');
INSERT INTO logged_model_params VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'test_size','0.2');
INSERT INTO logged_model_params VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'random_state','7');
INSERT INTO logged_model_params VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'n_estimators','1000');
INSERT INTO logged_model_params VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'learning_rate','0.1');
INSERT INTO logged_model_params VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'max_depth','6');
INSERT INTO logged_model_params VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'min_samples_leaf','9');
INSERT INTO logged_model_params VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'max_features','0.1');
INSERT INTO logged_model_params VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'test_size','0.2');
INSERT INTO logged_model_params VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'random_state','7');
INSERT INTO logged_model_params VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'n_estimators','1000');
INSERT INTO logged_model_params VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'learning_rate','0.1');
INSERT INTO logged_model_params VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'max_depth','6');
INSERT INTO logged_model_params VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'min_samples_leaf','9');
INSERT INTO logged_model_params VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'max_features','0.1');
INSERT INTO logged_model_params VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'test_size','0.2');
INSERT INTO logged_model_params VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'random_state','7');
CREATE TABLE logged_model_tags (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	tag_key VARCHAR(255) NOT NULL,
	tag_value TEXT NOT NULL,
	CONSTRAINT logged_model_tags_pk PRIMARY KEY (model_id, tag_key),
	CONSTRAINT fk_logged_model_tags_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_tags_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE
);
INSERT INTO logged_model_tags VALUES('m-381dfab095e04fd0985077c559c994e7',1,'mlflow.user','cheeyoungchang');
INSERT INTO logged_model_tags VALUES('m-381dfab095e04fd0985077c559c994e7',1,'mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py');
INSERT INTO logged_model_tags VALUES('m-381dfab095e04fd0985077c559c994e7',1,'mlflow.source.type','LOCAL');
INSERT INTO logged_model_tags VALUES('m-381dfab095e04fd0985077c559c994e7',1,'mlflow.source.git.commit','7c7096693079143753d2cfb938e9be23aa459873');
INSERT INTO logged_model_tags VALUES('m-381dfab095e04fd0985077c559c994e7',1,'mlflow.modelVersions','[{"name": "hdb-predictor", "version": 1}]');
INSERT INTO logged_model_tags VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'mlflow.user','cheeyoungchang');
INSERT INTO logged_model_tags VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py');
INSERT INTO logged_model_tags VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'mlflow.source.type','LOCAL');
INSERT INTO logged_model_tags VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'mlflow.source.git.commit','6e0bb5dbe913c22f91de7e3e0b4ffe32c6a7e425');
INSERT INTO logged_model_tags VALUES('m-f87f54e6757a48489e0f106fd862388c',1,'mlflow.modelVersions','[{"name": "hdb-predictor", "version": 2}]');
INSERT INTO logged_model_tags VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'mlflow.user','cheeyoungchang');
INSERT INTO logged_model_tags VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py');
INSERT INTO logged_model_tags VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'mlflow.source.type','LOCAL');
INSERT INTO logged_model_tags VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'mlflow.source.git.commit','6e0bb5dbe913c22f91de7e3e0b4ffe32c6a7e425');
INSERT INTO logged_model_tags VALUES('m-a2f7dc39cffa44f9b114328acfefdc0f',1,'mlflow.modelVersions','[{"name": "hdb-predictor", "version": 3}]');
INSERT INTO logged_model_tags VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'mlflow.user','cheeyoungchang');
INSERT INTO logged_model_tags VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'mlflow.source.name','/Users/cheeyoungchang/hdb-mlops-platform/src/training/train.py');
INSERT INTO logged_model_tags VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'mlflow.source.type','LOCAL');
INSERT INTO logged_model_tags VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'mlflow.source.git.commit','6e0bb5dbe913c22f91de7e3e0b4ffe32c6a7e425');
INSERT INTO logged_model_tags VALUES('m-bf10a6cce5bd49549963d2db305cbcfd',1,'mlflow.modelVersions','[{"name": "hdb-predictor", "version": 4}]');
INSERT INTO logged_model_tags VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'mlflow.user','cheeyoungchang');
INSERT INTO logged_model_tags VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'mlflow.source.name','/Users/cheeyoungchang/Projects/hdb-mlops-platform/src/training/train.py');
INSERT INTO logged_model_tags VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'mlflow.source.type','LOCAL');
INSERT INTO logged_model_tags VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'mlflow.source.git.commit','e21c833e6c790599418340ad536bdb451ac0935f');
INSERT INTO logged_model_tags VALUES('m-82a509e18d914655aa3e3e356e8ab7a5',1,'mlflow.modelVersions','[{"name": "hdb-predictor", "version": 5}]');
INSERT INTO logged_model_tags VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'mlflow.user','cheeyoungchang');
INSERT INTO logged_model_tags VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'mlflow.source.name','/Users/cheeyoungchang/Projects/hdb-mlops-platform/src/training/train.py');
INSERT INTO logged_model_tags VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'mlflow.source.type','LOCAL');
INSERT INTO logged_model_tags VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'mlflow.source.git.commit','e21c833e6c790599418340ad536bdb451ac0935f');
INSERT INTO logged_model_tags VALUES('m-cb9ee5b748924c029ee932c3f7c078ba',1,'mlflow.modelVersions','[{"name": "hdb-predictor", "version": 6}]');
INSERT INTO logged_model_tags VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'mlflow.user','cheeyoungchang');
INSERT INTO logged_model_tags VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'mlflow.source.name','/Users/cheeyoungchang/Projects/hdb-mlops-platform/src/training/train.py');
INSERT INTO logged_model_tags VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'mlflow.source.type','LOCAL');
INSERT INTO logged_model_tags VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'mlflow.source.git.commit','382cd68db0974ef9674dfd0ea818cc07ad5715d1');
INSERT INTO logged_model_tags VALUES('m-72a82cd491384c26811e7927af1ca6fd',1,'mlflow.modelVersions','[{"name": "hdb-predictor", "version": 7}]');
CREATE TABLE assessments (
	assessment_id VARCHAR(50) NOT NULL,
	trace_id VARCHAR(50) NOT NULL,
	name VARCHAR(250) NOT NULL,
	assessment_type VARCHAR(20) NOT NULL,
	value TEXT NOT NULL,
	error TEXT,
	created_timestamp BIGINT NOT NULL,
	last_updated_timestamp BIGINT NOT NULL,
	source_type VARCHAR(50) NOT NULL,
	source_id VARCHAR(250),
	run_id VARCHAR(32),
	span_id VARCHAR(50),
	rationale TEXT,
	overrides VARCHAR(50),
	valid BOOLEAN NOT NULL,
	assessment_metadata TEXT,
	CONSTRAINT assessments_pk PRIMARY KEY (assessment_id),
	CONSTRAINT fk_assessments_trace_id FOREIGN KEY(trace_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
);
CREATE TABLE spans (
	trace_id VARCHAR(50) NOT NULL,
	experiment_id INTEGER NOT NULL,
	span_id VARCHAR(50) NOT NULL,
	parent_span_id VARCHAR(50),
	name TEXT,
	type VARCHAR(500),
	status VARCHAR(50) NOT NULL,
	start_time_unix_nano BIGINT NOT NULL,
	end_time_unix_nano BIGINT,
	duration_ns BIGINT GENERATED ALWAYS AS (end_time_unix_nano - start_time_unix_nano) STORED,
	content TEXT NOT NULL, dimension_attributes JSON,
	CONSTRAINT spans_pk PRIMARY KEY (trace_id, span_id),
	CONSTRAINT fk_spans_trace_id FOREIGN KEY(trace_id) REFERENCES trace_info (request_id) ON DELETE CASCADE,
	CONSTRAINT fk_spans_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
);
CREATE TABLE entity_associations (
	association_id VARCHAR(36) NOT NULL,
	source_type VARCHAR(36) NOT NULL,
	source_id VARCHAR(36) NOT NULL,
	destination_type VARCHAR(36) NOT NULL,
	destination_id VARCHAR(36) NOT NULL,
	created_time BIGINT,
	CONSTRAINT entity_associations_pk PRIMARY KEY (source_type, source_id, destination_type, destination_id)
);
CREATE TABLE webhook_events (
	webhook_id VARCHAR(256) NOT NULL,
	entity VARCHAR(50) NOT NULL,
	action VARCHAR(50) NOT NULL,
	CONSTRAINT webhook_event_pk PRIMARY KEY (webhook_id, entity, action),
	FOREIGN KEY(webhook_id) REFERENCES webhooks (webhook_id) ON DELETE cascade
);
CREATE TABLE scorers (
	experiment_id INTEGER NOT NULL,
	scorer_name VARCHAR(256) NOT NULL,
	scorer_id VARCHAR(36) NOT NULL,
	CONSTRAINT scorer_pk PRIMARY KEY (scorer_id),
	CONSTRAINT fk_scorers_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE
);
CREATE TABLE scorer_versions (
	scorer_id VARCHAR(36) NOT NULL,
	scorer_version INTEGER NOT NULL,
	serialized_scorer TEXT NOT NULL,
	creation_time BIGINT,
	CONSTRAINT scorer_version_pk PRIMARY KEY (scorer_id, scorer_version),
	CONSTRAINT fk_scorer_versions_scorer_id FOREIGN KEY(scorer_id) REFERENCES scorers (scorer_id) ON DELETE CASCADE
);
CREATE TABLE evaluation_dataset_tags (
	dataset_id VARCHAR(36) NOT NULL,
	"key" VARCHAR(255) NOT NULL,
	value VARCHAR(5000),
	CONSTRAINT evaluation_dataset_tags_pk PRIMARY KEY (dataset_id, "key"),
	CONSTRAINT fk_evaluation_dataset_tags_dataset_id FOREIGN KEY(dataset_id) REFERENCES evaluation_datasets (dataset_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "evaluation_dataset_records" (
	dataset_record_id VARCHAR(36) NOT NULL,
	dataset_id VARCHAR(36) NOT NULL,
	inputs JSON NOT NULL,
	expectations JSON,
	tags JSON,
	source JSON,
	source_id VARCHAR(36),
	source_type VARCHAR(255),
	created_time BIGINT,
	last_update_time BIGINT,
	created_by VARCHAR(255),
	last_updated_by VARCHAR(255),
	input_hash VARCHAR(64) NOT NULL, outputs JSON,
	CONSTRAINT evaluation_dataset_records_pk PRIMARY KEY (dataset_record_id),
	CONSTRAINT fk_evaluation_dataset_records_dataset_id FOREIGN KEY(dataset_id) REFERENCES evaluation_datasets (dataset_id) ON DELETE CASCADE,
	CONSTRAINT unique_dataset_input UNIQUE (dataset_id, input_hash)
);
CREATE TABLE endpoint_model_mappings (
	mapping_id VARCHAR(36) NOT NULL,
	endpoint_id VARCHAR(36) NOT NULL,
	model_definition_id VARCHAR(36) NOT NULL,
	weight FLOAT NOT NULL,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL, linkage_type VARCHAR(64) DEFAULT 'PRIMARY' NOT NULL, fallback_order INTEGER,
	CONSTRAINT endpoint_model_mappings_pk PRIMARY KEY (mapping_id),
	CONSTRAINT fk_endpoint_model_mappings_endpoint_id FOREIGN KEY(endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE CASCADE,
	CONSTRAINT fk_endpoint_model_mappings_model_definition_id FOREIGN KEY(model_definition_id) REFERENCES model_definitions (model_definition_id)
);
CREATE TABLE endpoint_bindings (
	endpoint_id VARCHAR(36) NOT NULL,
	resource_type VARCHAR(50) NOT NULL,
	resource_id VARCHAR(255) NOT NULL,
	created_at BIGINT NOT NULL,
	created_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255), display_name VARCHAR(255),
	CONSTRAINT endpoint_bindings_pk PRIMARY KEY (endpoint_id, resource_type, resource_id),
	CONSTRAINT fk_endpoint_bindings_endpoint_id FOREIGN KEY(endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE CASCADE
);
CREATE TABLE endpoint_tags (
	"key" VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	endpoint_id VARCHAR(36) NOT NULL,
	CONSTRAINT endpoint_tag_pk PRIMARY KEY ("key", endpoint_id),
	CONSTRAINT fk_endpoint_tags_endpoint_id FOREIGN KEY(endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE CASCADE
);
CREATE TABLE trace_metrics (
	request_id VARCHAR(50) NOT NULL,
	"key" VARCHAR(250) NOT NULL,
	value FLOAT,
	CONSTRAINT trace_metrics_pk PRIMARY KEY (request_id, "key"),
	CONSTRAINT fk_trace_metrics_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
);
CREATE TABLE online_scoring_configs (
	online_scoring_config_id VARCHAR(36) NOT NULL,
	scorer_id VARCHAR(36) NOT NULL,
	sample_rate FLOAT NOT NULL,
	experiment_id INTEGER NOT NULL,
	filter_string TEXT,
	CONSTRAINT online_scoring_config_pk PRIMARY KEY (online_scoring_config_id),
	CONSTRAINT fk_online_scoring_configs_scorer_id FOREIGN KEY(scorer_id) REFERENCES scorers (scorer_id) ON DELETE CASCADE,
	CONSTRAINT fk_online_scoring_configs_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
);
CREATE TABLE span_metrics (
	trace_id VARCHAR(50) NOT NULL,
	span_id VARCHAR(50) NOT NULL,
	"key" VARCHAR(250) NOT NULL,
	value FLOAT,
	CONSTRAINT span_metrics_pk PRIMARY KEY (trace_id, span_id, "key"),
	CONSTRAINT fk_span_metrics_span FOREIGN KEY(trace_id, span_id) REFERENCES spans (trace_id, span_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "experiments" (
	experiment_id INTEGER NOT NULL,
	name VARCHAR(256) NOT NULL,
	artifact_location VARCHAR(256),
	lifecycle_stage VARCHAR(32),
	creation_time BIGINT,
	last_update_time BIGINT,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT experiment_pk PRIMARY KEY (experiment_id),
	CONSTRAINT experiments_lifecycle_stage CHECK (lifecycle_stage IN ('active', 'deleted')),
	CONSTRAINT uq_experiments_workspace_name UNIQUE (workspace, name)
);
INSERT INTO experiments VALUES(0,'Default','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/0','active',1777345885790,1777345885790,'default');
INSERT INTO experiments VALUES(1,'hdb-resale-price','/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1','active',1777345885793,1777345885793,'default');
CREATE TABLE IF NOT EXISTS "registered_models" (
	name VARCHAR(256) NOT NULL,
	creation_time BIGINT,
	last_updated_time BIGINT,
	description VARCHAR(5000),
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT registered_model_pk PRIMARY KEY (workspace, name)
);
INSERT INTO registered_models VALUES('hdb-predictor',1777346482486,1781141597585,NULL,'default');
CREATE TABLE IF NOT EXISTS "model_versions" (
	name VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	creation_time BIGINT,
	last_updated_time BIGINT,
	description VARCHAR(5000),
	user_id VARCHAR(256),
	current_stage VARCHAR(20),
	source VARCHAR(500),
	run_id VARCHAR(32),
	status VARCHAR(20),
	status_message VARCHAR(500),
	run_link VARCHAR(500),
	storage_location VARCHAR(500),
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT model_version_pk PRIMARY KEY (workspace, name, version),
	CONSTRAINT fk_model_versions_registered_models FOREIGN KEY(workspace, name) REFERENCES registered_models (workspace, name) ON UPDATE CASCADE
);
INSERT INTO model_versions VALUES('hdb-predictor',1,1777346482495,1777346482495,NULL,NULL,'None','models:/m-381dfab095e04fd0985077c559c994e7','b90b186495804fb69cb246143d047dcb','READY',NULL,NULL,'/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-381dfab095e04fd0985077c559c994e7/artifacts','default');
INSERT INTO model_versions VALUES('hdb-predictor',2,1777440901731,1777440901731,NULL,NULL,'None','models:/m-f87f54e6757a48489e0f106fd862388c','daaef3393bc64e6aacc666b46fa35faa','READY',NULL,NULL,'/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-f87f54e6757a48489e0f106fd862388c/artifacts','default');
INSERT INTO model_versions VALUES('hdb-predictor',3,1777441987388,1777441987388,NULL,NULL,'None','models:/m-a2f7dc39cffa44f9b114328acfefdc0f','2ec9ef9f05f84e5fa7a31c4c7138f34a','READY',NULL,NULL,'/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-a2f7dc39cffa44f9b114328acfefdc0f/artifacts','default');
INSERT INTO model_versions VALUES('hdb-predictor',4,1777444220109,1777444220109,NULL,NULL,'None','models:/m-bf10a6cce5bd49549963d2db305cbcfd','d917d0feffd3470894512d9f759e755d','READY',NULL,NULL,'/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-bf10a6cce5bd49549963d2db305cbcfd/artifacts','default');
INSERT INTO model_versions VALUES('hdb-predictor',5,1777949303321,1777949303321,NULL,NULL,'None','models:/m-82a509e18d914655aa3e3e356e8ab7a5','a08f4fd052694ab79cd28b5111ba7fb9','READY',NULL,NULL,'/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-82a509e18d914655aa3e3e356e8ab7a5/artifacts','default');
INSERT INTO model_versions VALUES('hdb-predictor',6,1777951561947,1777951561947,NULL,NULL,'None','models:/m-cb9ee5b748924c029ee932c3f7c078ba','408da2e13b794af59086d35fca34e030','READY',NULL,NULL,'/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-cb9ee5b748924c029ee932c3f7c078ba/artifacts','default');
INSERT INTO model_versions VALUES('hdb-predictor',7,1781141597585,1781141597585,NULL,NULL,'None','models:/m-72a82cd491384c26811e7927af1ca6fd','26b45ff54a734389ae23eb2ea8b7ab68','READY',NULL,NULL,'/Users/cheeyoungchang/hdb-mlops-platform/mlruns/1/models/m-72a82cd491384c26811e7927af1ca6fd/artifacts','default');
CREATE TABLE IF NOT EXISTS "registered_model_tags" (
	"key" VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	name VARCHAR(256) NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT registered_model_tag_pk PRIMARY KEY (workspace, "key", name),
	CONSTRAINT fk_registered_model_tags_registered_models FOREIGN KEY(workspace, name) REFERENCES registered_models (workspace, name) ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "model_version_tags" (
	"key" VARCHAR(250) NOT NULL,
	value TEXT,
	name VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT model_version_tag_pk PRIMARY KEY (workspace, "key", name, version),
	CONSTRAINT fk_model_version_tags_model_versions FOREIGN KEY(workspace, name, version) REFERENCES model_versions (workspace, name, version) ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "registered_model_aliases" (
	alias VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	name VARCHAR(256) NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT registered_model_alias_pk PRIMARY KEY (workspace, name, alias),
	CONSTRAINT fk_registered_model_aliases_registered_models FOREIGN KEY(workspace, name) REFERENCES registered_models (workspace, name) ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO registered_model_aliases VALUES('champion',7,'hdb-predictor','default');
CREATE TABLE IF NOT EXISTS "evaluation_datasets" (
	dataset_id VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	schema TEXT,
	profile TEXT,
	digest VARCHAR(64),
	created_time BIGINT,
	last_update_time BIGINT,
	created_by VARCHAR(255),
	last_updated_by VARCHAR(255),
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT evaluation_datasets_pk PRIMARY KEY (dataset_id)
);
CREATE TABLE IF NOT EXISTS "webhooks" (
	webhook_id VARCHAR(256) NOT NULL,
	name VARCHAR(256) NOT NULL,
	description VARCHAR(1000),
	url VARCHAR(500) NOT NULL,
	status VARCHAR(20) DEFAULT 'ACTIVE' NOT NULL,
	secret VARCHAR(1000),
	creation_timestamp BIGINT,
	last_updated_timestamp BIGINT,
	deleted_timestamp BIGINT,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT webhook_pk PRIMARY KEY (webhook_id)
);
CREATE TABLE IF NOT EXISTS "secrets" (
	secret_id VARCHAR(36) NOT NULL,
	secret_name VARCHAR(255) NOT NULL,
	encrypted_value BLOB NOT NULL,
	wrapped_dek BLOB NOT NULL,
	kek_version INTEGER NOT NULL,
	masked_value VARCHAR(500) NOT NULL,
	provider VARCHAR(64),
	auth_config TEXT,
	description TEXT,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT secrets_pk PRIMARY KEY (secret_id),
	CONSTRAINT uq_secrets_workspace_secret_name UNIQUE (workspace, secret_name)
);
CREATE TABLE IF NOT EXISTS "endpoints" (
	endpoint_id VARCHAR(36) NOT NULL,
	name VARCHAR(255),
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	routing_strategy VARCHAR(64),
	fallback_config_json TEXT,
	experiment_id INTEGER,
	usage_tracking BOOLEAN DEFAULT '0' NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT endpoints_pk PRIMARY KEY (endpoint_id),
	CONSTRAINT fk_endpoints_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE SET NULL,
	CONSTRAINT uq_endpoints_workspace_name UNIQUE (workspace, name)
);
CREATE TABLE IF NOT EXISTS "model_definitions" (
	model_definition_id VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	secret_id VARCHAR(36),
	provider VARCHAR(64) NOT NULL,
	model_name VARCHAR(256) NOT NULL,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT model_definitions_pk PRIMARY KEY (model_definition_id),
	CONSTRAINT fk_model_definitions_secret_id FOREIGN KEY(secret_id) REFERENCES secrets (secret_id) ON DELETE SET NULL,
	CONSTRAINT uq_model_definitions_workspace_name UNIQUE (workspace, name)
);
CREATE TABLE IF NOT EXISTS "jobs" (
	id VARCHAR(36) NOT NULL,
	creation_time BIGINT NOT NULL,
	job_name VARCHAR(500) NOT NULL,
	params TEXT NOT NULL,
	timeout FLOAT,
	status INTEGER NOT NULL,
	result TEXT,
	retry_count INTEGER NOT NULL,
	last_update_time BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL, status_details JSON,
	CONSTRAINT jobs_pk PRIMARY KEY (id)
);
CREATE TABLE workspaces (
	name VARCHAR(63) NOT NULL,
	description TEXT,
	default_artifact_root TEXT, trace_archival_location TEXT, trace_archival_retention VARCHAR(32),
	CONSTRAINT workspaces_pk PRIMARY KEY (name)
);
INSERT INTO workspaces VALUES('default','Default workspace for legacy resources',NULL,NULL,NULL);
CREATE TABLE budget_policies (
	budget_policy_id VARCHAR(36) NOT NULL,
	budget_unit VARCHAR(32) NOT NULL,
	budget_amount FLOAT NOT NULL,
	duration_unit VARCHAR(32) NOT NULL,
	duration_value INTEGER NOT NULL,
	target_scope VARCHAR(32) NOT NULL,
	budget_action VARCHAR(32) NOT NULL,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT budget_policies_pk PRIMARY KEY (budget_policy_id)
);
CREATE TABLE issues (
	issue_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(250) NOT NULL,
	description TEXT NOT NULL,
	status VARCHAR(50) NOT NULL,
	severity VARCHAR(50),
	root_causes TEXT,
	source_run_id VARCHAR(32),
	categories TEXT,
	created_timestamp BIGINT NOT NULL,
	last_updated_timestamp BIGINT NOT NULL,
	created_by VARCHAR(255),
	CONSTRAINT issues_pk PRIMARY KEY (issue_id),
	CONSTRAINT fk_issues_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE,
	CONSTRAINT fk_issues_source_run_id FOREIGN KEY(source_run_id) REFERENCES runs (run_uuid) ON DELETE SET NULL
);
CREATE TABLE guardrails (
	guardrail_id VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	scorer_id VARCHAR(36) NOT NULL,
	scorer_version INTEGER NOT NULL,
	stage VARCHAR(32) NOT NULL,
	action VARCHAR(32) NOT NULL,
	action_endpoint_id VARCHAR(36),
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT guardrails_pk PRIMARY KEY (guardrail_id),
	CONSTRAINT fk_guardrails_scorer_version FOREIGN KEY(scorer_id, scorer_version) REFERENCES scorer_versions (scorer_id, scorer_version),
	CONSTRAINT fk_guardrails_action_endpoint_id FOREIGN KEY(action_endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE SET NULL
);
CREATE TABLE guardrail_configs (
	endpoint_id VARCHAR(36) NOT NULL,
	guardrail_id VARCHAR(36) NOT NULL,
	execution_order INTEGER,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT guardrail_configs_pk PRIMARY KEY (endpoint_id, guardrail_id),
	CONSTRAINT fk_guardrail_configs_endpoint_id FOREIGN KEY(endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE CASCADE,
	CONSTRAINT fk_guardrail_configs_guardrail_id FOREIGN KEY(guardrail_id) REFERENCES guardrails (guardrail_id) ON DELETE CASCADE
);
CREATE TRIGGER prevent_secrets_aad_mutation
BEFORE UPDATE ON secrets
FOR EACH ROW
WHEN OLD.secret_id != NEW.secret_id OR OLD.secret_name != NEW.secret_name
BEGIN
    SELECT RAISE(ABORT, 'secret_id and secret_name are immutable (used as AAD in encryption)');
END;
CREATE INDEX index_metrics_run_uuid ON metrics (run_uuid);
CREATE INDEX index_latest_metrics_run_uuid ON latest_metrics (run_uuid);
CREATE INDEX index_inputs_input_uuid ON inputs (input_uuid);
CREATE INDEX index_inputs_destination_type_destination_id_source_type ON inputs (destination_type, destination_id, source_type);
CREATE INDEX index_params_run_uuid ON params (run_uuid);
CREATE INDEX index_trace_info_experiment_id_timestamp_ms ON trace_info (experiment_id, timestamp_ms);
CREATE INDEX index_trace_tags_request_id ON trace_tags (request_id);
CREATE INDEX index_trace_request_metadata_request_id ON trace_request_metadata (request_id);
CREATE INDEX index_tags_run_uuid ON tags (run_uuid);
CREATE INDEX index_datasets_experiment_id_dataset_source_type ON datasets (experiment_id, dataset_source_type);
CREATE INDEX index_datasets_dataset_uuid ON datasets (dataset_uuid);
CREATE INDEX index_logged_model_metrics_model_id ON logged_model_metrics (model_id);
CREATE INDEX index_assessments_trace_id_created_timestamp ON assessments (trace_id, created_timestamp);
CREATE INDEX index_assessments_run_id_created_timestamp ON assessments (run_id, created_timestamp);
CREATE INDEX index_assessments_last_updated_timestamp ON assessments (last_updated_timestamp);
CREATE INDEX index_assessments_assessment_type ON assessments (assessment_type);
CREATE INDEX index_spans_experiment_id ON spans (experiment_id);
CREATE INDEX index_spans_experiment_id_status_type ON spans (experiment_id, status, type);
CREATE INDEX index_spans_experiment_id_type_status ON spans (experiment_id, type, status);
CREATE INDEX index_spans_experiment_id_duration ON spans (experiment_id, duration_ns);
CREATE INDEX index_entity_associations_association_id ON entity_associations (association_id);
CREATE INDEX index_entity_associations_reverse_lookup ON entity_associations (destination_type, destination_id, source_type, source_id);
CREATE INDEX idx_webhook_events_entity ON webhook_events (entity);
CREATE INDEX idx_webhook_events_action ON webhook_events (action);
CREATE INDEX idx_webhook_events_entity_action ON webhook_events (entity, action);
CREATE UNIQUE INDEX index_scorers_experiment_id_scorer_name ON scorers (experiment_id, scorer_name);
CREATE INDEX index_scorer_versions_scorer_id ON scorer_versions (scorer_id);
CREATE INDEX index_evaluation_dataset_tags_dataset_id ON evaluation_dataset_tags (dataset_id);
CREATE INDEX index_evaluation_dataset_records_dataset_id ON evaluation_dataset_records (dataset_id);
CREATE INDEX index_endpoint_model_mappings_endpoint_id ON endpoint_model_mappings (endpoint_id);
CREATE INDEX index_endpoint_model_mappings_model_definition_id ON endpoint_model_mappings (model_definition_id);
CREATE INDEX index_endpoint_tags_endpoint_id ON endpoint_tags (endpoint_id);
CREATE INDEX index_trace_metrics_request_id ON trace_metrics (request_id);
CREATE UNIQUE INDEX unique_endpoint_model_linkage_mapping ON endpoint_model_mappings (endpoint_id, model_definition_id, linkage_type);
CREATE INDEX index_span_metrics_trace_id_span_id ON span_metrics (trace_id, span_id);
CREATE INDEX index_evaluation_datasets_created_time ON evaluation_datasets (created_time);
CREATE INDEX index_evaluation_datasets_name ON evaluation_datasets (name);
CREATE INDEX idx_webhooks_name ON webhooks (name);
CREATE INDEX idx_webhooks_status ON webhooks (status);
CREATE INDEX index_model_definitions_provider ON model_definitions (provider);
CREATE INDEX index_model_definitions_secret_id ON model_definitions (secret_id);
CREATE INDEX index_jobs_name_status_creation_time ON jobs (job_name, workspace, status, creation_time);
CREATE INDEX idx_experiments_workspace ON experiments (workspace);
CREATE INDEX idx_registered_models_workspace ON registered_models (workspace);
CREATE INDEX idx_experiments_workspace_creation_time ON experiments (workspace, creation_time);
CREATE INDEX idx_evaluation_datasets_workspace ON evaluation_datasets (workspace);
CREATE INDEX idx_webhooks_workspace ON webhooks (workspace);
CREATE INDEX idx_secrets_workspace ON secrets (workspace);
CREATE INDEX idx_endpoints_workspace ON endpoints (workspace);
CREATE INDEX idx_model_definitions_workspace ON model_definitions (workspace);
CREATE INDEX idx_budget_policies_workspace ON budget_policies (workspace);
CREATE INDEX index_issues_experiment_id ON issues (experiment_id);
CREATE INDEX index_issues_source_run_id ON issues (source_run_id);
CREATE INDEX index_issues_status ON issues (status);
CREATE INDEX index_metrics_run_uuid_key_step ON metrics (run_uuid, "key", step);
CREATE INDEX idx_guardrails_workspace ON guardrails (workspace);
CREATE INDEX idx_guardrails_scorer ON guardrails (scorer_id, scorer_version);
CREATE INDEX idx_guardrail_configs_endpoint_id ON guardrail_configs (endpoint_id);
CREATE INDEX idx_guardrail_configs_guardrail_id ON guardrail_configs (guardrail_id);
CREATE INDEX idx_model_version_tags_workspace_name_version ON model_version_tags (workspace, name, version);
CREATE INDEX idx_registered_model_tags_workspace_name ON registered_model_tags (workspace, name);
COMMIT;
