# CHANGELOG



## v0.16.0 (2026-07-16)

### Feature

* feat: Bump huggingface_hub to support DINOv3 (#13)

* build: strip v prefix from release docker tags

Normalize semantic-release versions before docker push so image tags publish as mbari/vss:0.1.2 instead of mbari/vss:v0.1.2.

Made-with: Cursor

* increase performance with batch

* perf: improve batching

* fix: skip corrupt images during embedding and report failed filenames

Handle truncated or unreadable images without failing the whole batch, and return failed_filenames in embedding job results.

Co-authored-by: Cursor &lt;cursoragent@cursor.com&gt;

---------

Co-authored-by: Cursor &lt;cursoragent@cursor.com&gt; ([`52acb24`](https://github.com/mbari-org/vss/commit/52acb2489acdd32c2cd366ad7464b2d8625dd81c))

### Unknown

* Add non-root user to all Docker images (#11)

* Initial plan

* Add non-root user (1001:1001) to Dockerfile and Dockerfile.cuda with overridable ARG

Agent-Logs-Url: https://github.com/mbari-org/vss/sessions/81fed97a-54da-4d92-aa46-3c7802b12681

Co-authored-by: danellecline &lt;1424813+danellecline@users.noreply.github.com&gt;

---------

Co-authored-by: copilot-swe-agent[bot] &lt;198982749+Copilot@users.noreply.github.com&gt;
Co-authored-by: danellecline &lt;1424813+danellecline@users.noreply.github.com&gt; ([`6784c22`](https://github.com/mbari-org/vss/commit/6784c222ead3fe5bb673eb4a5a3351629882c54c))


## v0.15.0 (2026-03-26)

### Build

* build: smaller base cuda image for multi-stage ([`8658a17`](https://github.com/mbari-org/vss/commit/8658a17b6e298c7ac61d6a7df95373440a0eb797))

* build: cuda 13 build with ENV DEBIAN_FRONTEND=noninteractive ([`502ec75`](https://github.com/mbari-org/vss/commit/502ec750efb01f8781c89b2624aa69fadf92d5c2))

* build: cuda 13 build ([`d26850f`](https://github.com/mbari-org/vss/commit/d26850f374f5f2b70d9b783f4cfc784d20bffa29))

### Feature

* feat(fiftyone): bump to latest sync version v0.1.0 ([`98dac22`](https://github.com/mbari-org/vss/commit/98dac224adda221709d9cc0c1a24ae072ef96f00))


## v0.14.1 (2026-03-12)

### Build

* build: added missing IN_DOCKER to cuda build ([`e587417`](https://github.com/mbari-org/vss/commit/e587417754318573eda469d7ef3f5fd8ceac92b1))

### Fix

* fix: handle spooled and raw bytes ([`3a01d72`](https://github.com/mbari-org/vss/commit/3a01d720f81b620b88f0439ca08556ac6a87cba8))


## v0.14.0 (2026-03-12)

### Feature

* feat: added CORSMiddleware ([`0719845`](https://github.com/mbari-org/vss/commit/07198450b81400a33e7706b128b3af332f7e95ea))


## v0.13.1 (2026-03-10)

### Build

* build: renamed docker images to vss ([`45c17f8`](https://github.com/mbari-org/vss/commit/45c17f892cd744c3af139e629fde4272277bf4ec))

### Documentation

* docs: added documentation on justfile and better compose.yml dependency ([`a5292d4`](https://github.com/mbari-org/vss/commit/a5292d482c72b6d5f7646807d479c874bc1ddd41))

* docs: added documentation on justfile and better compose.yml dependency ([`11ffc70`](https://github.com/mbari-org/vss/commit/11ffc7061ba6db136a8e2eb28bc5114868e61957))

### Performance

* perf: allow for larger ws-max-size in uvicorn command for embedding support ([`aaeb795`](https://github.com/mbari-org/vss/commit/aaeb7952b02fafa590a44873c118fe3678bded19))


## v0.13.0 (2026-03-06)

### Feature

* feat: add web socket support and better handling of model weights

* testing  logging and byte fix
* correct cuda build
* add check for existence of model path if local
* web socket, tests, and better handling of model weights ([`a896f50`](https://github.com/mbari-org/vss/commit/a896f50e59eea222409f8c659ed09755d487e4a5))


## v0.12.2 (2026-03-03)

### Build

* build: reduce docker image size by using multi-stage and removing unnecessary files ([`8991489`](https://github.com/mbari-org/vss/commit/8991489ddd3aedcf364627ca2959d6ebad9a25c5))

### Fix

* fix: move inputs to same device as model ([`636619d`](https://github.com/mbari-org/vss/commit/636619d4c0b0297540ac52afb19f7dd65042ce2c))


## v0.12.1 (2026-02-28)

### Build

* build: updated with missing CONFIG_PATH ([`5031be1`](https://github.com/mbari-org/vss/commit/5031be1c30edecd882bbfb5c7e77f6de4702e6b6))

### Documentation

* docs: updated with latest REST api screenshot ([`433c176`](https://github.com/mbari-org/vss/commit/433c176fb7be3b1572132df4f8ba30a2e8eb87e8))

### Fix

* fix: suppress pynvml warning when using nvidia, pin requirements.txt with best version for cross-platform use with redis ([`c68c7b8`](https://github.com/mbari-org/vss/commit/c68c7b86695bc1dcd8a161f1ad5bf64fe4c029d4))


## v0.12.0 (2026-02-23)

### Documentation

* docs: shorten README.md ([`9ba45cc`](https://github.com/mbari-org/vss/commit/9ba45cc0171793138c13e4b01b88bae0c7ac0e01))

### Feature

* feat: add support for returning embeddings (#8)

This mostly adds a new endpoint /embed/{project}  that returns raw ViT embeddings for uploaded images (instead of KNN predictions) along with proper testing of the endpoing.

A few simplifications were made to build and CI including:

CONFIG_PATH is no longer a required environment variable. It now defaults to &lt;project_root&gt;/config 
Renamed redis container to redis-stack which more clearly reflects the full redis stack capability needed for vector support.

Some clean-up of unused exclusions in precommit files, improved README.md documentation on configuration and justfile recipes for convenience. ([`8658984`](https://github.com/mbari-org/vss/commit/86589840ba635e9417bf00214c1a40dc352204ac))


## v0.11.5 (2026-02-14)

### Documentation

* docs: added worfklow and refresher on README.md ([`b3d7960`](https://github.com/mbari-org/vss/commit/b3d7960267099d5b6487fd9dfcfc9903da424f87))

### Fix

* fix: remove job comment about time results will be available as this may be indefinite, depending on the deployment ([`43477d1`](https://github.com/mbari-org/vss/commit/43477d14dc99e85efe25b883869f6e66c2842091))


## v0.11.4 (2026-01-29)

### Performance

* perf: switch to nanosecond log ([`136d6e7`](https://github.com/mbari-org/vss/commit/136d6e74ec20d6df898de18f12d977c14642f689))


## v0.11.3 (2026-01-16)

### Performance

* perf: limit /ids/{project} endpoint to return first 100 IDs (#5)

Limit the endpoint to the first 100 IDs as this is just a convenience endpoint to checking the database ([`65632ee`](https://github.com/mbari-org/vss/commit/65632ee8bc6fe79ffef018dfb4ab710b7b4a03f6))

### Unknown

* Feat/config (#4)

* add url based loading of config with 
* added caching of config
* add docker check for internal versus external URL ([`51422ea`](https://github.com/mbari-org/vss/commit/51422ea4799d29b48999a27e8125399c20b9a48d))


## v0.11.2 (2025-10-11)

### Performance

* perf: significant speed-up in preprocessing and replaced outdated pynvml with nvidia-ml-py ([`a7fd52d`](https://github.com/mbari-org/vss/commit/a7fd52db9fad66af4abccc05e75c8e5b8b354434))


## v0.11.1 (2025-09-15)

### Documentation

* docs: clean-up of README for clarity and add more badges, author. ([`e745171`](https://github.com/mbari-org/vss/commit/e7451711ae1354493b43eded1e2408441bf6a2d0))

### Fix

* fix: correct semantic release version ([`097a87f`](https://github.com/mbari-org/vss/commit/097a87f9204aaa0cf4fbcd6b9f9ba7255734ccbc))


## v0.11.0 (2025-09-14)

### Feature

* feat: export metrics via Prometheus ([`9aada8a`](https://github.com/mbari-org/vss/commit/9aada8a769d3e45f1f11f67aceacd226484897db))

### Unknown

* revert: remove connection pool which was a breaking change ([`075e8f7`](https://github.com/mbari-org/vss/commit/075e8f7fecda00c0d1d5df43a20efecc5e5c3ae1))


## v0.10.4 (2025-09-09)

### Performance

* perf: added connection pool ([`ebf9f7b`](https://github.com/mbari-org/vss/commit/ebf9f7bb46c6312dedc64c86d1222654788a250e))


## v0.10.3 (2025-09-09)

### Performance

* perf: cuDNN optimizations ([`237c496`](https://github.com/mbari-org/vss/commit/237c496d8f266f3b98cdf2ba35daebf792dc0729))


## v0.10.2 (2025-07-22)

### Fix

* fix: per project config and renaming output_dir to output_path ([`314d282`](https://github.com/mbari-org/vss/commit/314d282c18dfd75b36b59f630f732b6e779b63d7))


## v0.10.1 (2025-07-22)

### Fix

* fix: trigger release with updated GITHUB_TOKEN ([`1624898`](https://github.com/mbari-org/vss/commit/1624898de54fffc6f185e0861a511227b58d24b7))


## v0.10.0 (2025-07-22)

### Feature

* feat: return json output with path ([`6d2dd87`](https://github.com/mbari-org/vss/commit/6d2dd870efa935120dd025d991c3e6e841f57123))


## v0.9.1 (2025-07-20)

### Documentation

* docs: updated related work to more relevant repo ([`bd3b798`](https://github.com/mbari-org/vss/commit/bd3b79853415fc2c2aac384efff2027197af9bae))

### Fix

* fix: return location of json output correctly and message about timeout ([`09019d1`](https://github.com/mbari-org/vss/commit/09019d1f3390fe5e39e77aa0444dec56cab871b9))

* fix: GPU spawn for safe multiprocessing ([`7003888`](https://github.com/mbari-org/vss/commit/70038887020bfa8df66c5a778d5ee3f043782475))


## v0.9.0 (2025-07-07)

### Build

* build: correct tag for cuda build ([`139b4d0`](https://github.com/mbari-org/vss/commit/139b4d0399a71b26b01166061ce45829dd84bf59))

* build: remove arm64 build for cuda build ([`ab31f13`](https://github.com/mbari-org/vss/commit/ab31f1342e8e991395f9bb3cb04bf9704e1999ba))

### Feature

* feat: trigger release ([`84049b1`](https://github.com/mbari-org/vss/commit/84049b1feb261d659026445b0cb24cdd0dfa7918))

### Performance

* perf: add support for task execution through redis queue workers (#2)

This is a significant enhancement that adds:

Redis queue workers
Saves results to JSON output to a data mount
Multiple GPU execution for speed-up
Better logging
Tests for most endpoints
Health check for FastAPI-vss service ([`d6cc7ba`](https://github.com/mbari-org/vss/commit/d6cc7bac1113841eb1cebe7fd6085d6c49853300))


## v0.8.0 (2025-06-03)

### Feature

* feat: bump to mbari/aidata:1.55.1 ([`5efc603`](https://github.com/mbari-org/vss/commit/5efc60399a57c69a1363f6ff26b88794be3aca71))


## v0.7.0 (2025-06-02)

### Feature

* feat: bump to mbari/aidata:1.53.0 and try cached build ([`aa35945`](https://github.com/mbari-org/vss/commit/aa35945402b5ee06848f1b951b83236c3a1d12be))


## v0.6.0 (2025-06-01)

### Build

* build: updated release gh action ([`26b2a72`](https://github.com/mbari-org/vss/commit/26b2a722e90dba75227b04fc19a3373046faf3fd))

### Feature

* feat: only added doc, but triggering release workflow for test ([`80cc0f9`](https://github.com/mbari-org/vss/commit/80cc0f90a16a0e0093e15bac85bcf17011d9646d))


## v0.5.0 (2025-06-01)

### Build

* build: ignore code from base image ([`09d4955`](https://github.com/mbari-org/vss/commit/09d49552e3ac72efb3acb6af34f05ac56b4a1b85))

### Documentation

* docs: remove installation ([`401edf1`](https://github.com/mbari-org/vss/commit/401edf10c95cef888a23dbe62414ea8c60fc1efd))

* docs: minor rearranging ([`0a1b69d`](https://github.com/mbari-org/vss/commit/0a1b69dbcf5103215358c836569a8aad98345b8d))

* docs: updated README.md with features and icons consistent with other docs ([`9a1e470`](https://github.com/mbari-org/vss/commit/9a1e4702ac7083a72cd68a7f8f2467a02dcc2f34))

### Feature

* feat: bumped to aidata:1.53.0 ([`5cbed72`](https://github.com/mbari-org/vss/commit/5cbed72434e5246a12cd85fe168d3cfca7b270f4))


## v0.4.6 (2025-06-01)

### Build

* build: bumped base to aidata:1.41.9 and added cuda124 to just recipe ([`47d5d52`](https://github.com/mbari-org/vss/commit/47d5d5295593737325520625e385cee742a75181))

* build: bumped docker base and added recipe for docker build and push with provenance and SBOM ([`ea525dd`](https://github.com/mbari-org/vss/commit/ea525dd02c19f8a8028603374abee662b413e474))

### Documentation

* docs: updated restup image to link ([`33eba86`](https://github.com/mbari-org/vss/commit/33eba8618770ffeed7ac110a9648e88e52deedcb))

* docs: updated ui image ([`fdde441`](https://github.com/mbari-org/vss/commit/fdde441df62a9ae7dba40574aa92bab482096930))

### Performance

* perf: allow BATCH_SIZE setting through environment variable BATCH_SIZE for optimized GPU performance; default 32 ([`8c259ee`](https://github.com/mbari-org/vss/commit/8c259ee09722375db87839a3ee21aad7c9243f2d))


## v0.4.5 (2024-10-28)

### Performance

* perf: added GPU monitoring and pass through number of workers for improved redis performance ([`cfe8f6d`](https://github.com/mbari-org/vss/commit/cfe8f6ddc4bc36b3f6377c7b95349661866bc112))


## v0.4.4 (2024-09-30)

### Performance

* perf: added garbage collection, increase batch size and multiple workers for higher throughput ([`bad00b9`](https://github.com/mbari-org/vss/commit/bad00b9403738d64fa70e1b7e14aedb02d9382ad))


## v0.4.3 (2024-09-30)

### Performance

* perf: default to cuda ([`b8e14a5`](https://github.com/mbari-org/vss/commit/b8e14a53263723381446b71136c7768462b5036c))


## v0.4.2 (2024-09-30)

### Fix

* fix: handle any prediction exception ([`64e8f96`](https://github.com/mbari-org/vss/commit/64e8f96ef4a24fdb9acb356efd6f1d6f7287df6e))


## v0.4.1 (2024-09-16)

### Fix

* fix: bump version and updated docs with latest ui screen capture ([`2399296`](https://github.com/mbari-org/vss/commit/23992969ce70c7d819bb7a3e4689f4f3f567ee23))


## v0.4.0 (2024-09-16)

### Feature

* feat: switch to vss project config name ([`be4d0bb`](https://github.com/mbari-org/vss/commit/be4d0bb669711b907cee792f47659b448a056728))


## v0.3.1 (2024-09-06)

### Fix

* fix: fix typo ([`b5c9735`](https://github.com/mbari-org/vss/commit/b5c9735dbde1b34276b3799fbdb8338ce44fc9e9))


## v0.3.0 (2024-09-06)

### Feature

* feat: add support for different models ([`53743ff`](https://github.com/mbari-org/vss/commit/53743ff9b01a8bc0d038943ab999fc2f68f4fb3d))


## v0.2.1 (2024-09-04)

### Performance

* perf: bump to better indexing ([`7beb4b7`](https://github.com/mbari-org/vss/commit/7beb4b72a08186ca89f540262f054904509db9c0))


## v0.2.0 (2024-08-28)

### Feature

* feat: change to external config_path via environment variable CONFIG_PATH ([`352dc6b`](https://github.com/mbari-org/vss/commit/352dc6bddfa5fa4b3fe0656324ef63bd53d05251))


## v0.1.4 (2024-08-28)

### Fix

* fix: trigger CUDA release with updated docker build ([`dfb6838`](https://github.com/mbari-org/vss/commit/dfb68381d8a7ea445394e2dc53aae11ea1cd6c95))


## v0.1.3 (2024-08-28)

### Fix

* fix: add missing dependency ([`5f0c861`](https://github.com/mbari-org/vss/commit/5f0c86100caabc7f2e771a9f7dde4e4e87e9eca2))

### Unknown

* Update README.md ([`1a1b260`](https://github.com/mbari-org/vss/commit/1a1b2609f4c663adb8bbc9b5c6e4b59c40fded7e))


## v0.1.2 (2024-08-22)

### Fix

* fix: trigger release to update __init__.py ([`65e3ecd`](https://github.com/mbari-org/vss/commit/65e3ecdde528fb222bc349a9cae6c3c7c632f0a8))


## v0.1.1 (2024-08-12)

### Fix

* fix: correct handling of .env ([`17c2de0`](https://github.com/mbari-org/vss/commit/17c2de03d4bab636364a2def0399202b73de50b1))


## v0.1.0 (2024-07-30)

### Documentation

* docs: minor fix to link ([`f86a56a`](https://github.com/mbari-org/vss/commit/f86a56acbf2022420f98b0ea2e33aa7a5c352265))

### Feature

* feat: added project and knn api and aidata submodule ([`dc736e1`](https://github.com/mbari-org/vss/commit/dc736e181c18818a8f73131c2bfc81cdd27e7088))

* feat: added aidata submodule ([`ac509e1`](https://github.com/mbari-org/vss/commit/ac509e19618b15b3489f6e4ec229097adabacf06))

* feat: initial commit ([`ae974a4`](https://github.com/mbari-org/vss/commit/ae974a4aeeb5e2084b46ff27e2d18513708fff23))
