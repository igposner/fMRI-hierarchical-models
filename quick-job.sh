#!/bin/bash
srun \
--job-name=interactive \
--account=benjamin_kay \
--partition=tier1_cpu \
--oversubscribe \
--nodes=1 \
--ntasks-per-node=1 \
--cpus-per-task=1 \
--mem=8GB \
--time=8:00:00 \
--pty bash

# Launch a quick interactive job.
