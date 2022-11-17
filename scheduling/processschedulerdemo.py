import processscheduler as ps

# total number of jobs
n = 10

# job durations
p = [3, 2, 5, 4, 2, 3, 4, 2, 4, 6]

# number of resources [c1, c2] required per for each job
u = [[5, 1], [0, 4], [1, 4], [1, 3], [3, 2], [3, 1], [2, 4], [4, 0], [5, 2], [2, 5]]
# number of resources c and c2
c = [6, 8]

# job dependencies/precedences
S = [
    [1, 4],
    [1, 5],
    [2, 9],
    [2, 10],
    [3, 8],
    [4, 6],
    [4, 7],
    [5, 9],
    [5, 10],
    [6, 8],
    [6, 9],
    [7, 8],
]

pb = ps.SchedulingProblem("ResourceConstrainedProject")
# create n jobs
jobs = [ps.FixedDurationTask("Job_%i" % (i + 1), duration=p[i]) for i in range(n)]
# create the resources
C1_Ress = [ps.Worker("C1_Ress_%i" % i) for i in range(c[0])]
C2_Ress = [ps.Worker("C2_Ress_%i" % i) for i in range(c[1])]
# job precedences
for index_job_before, index_job_after in S:
    ps.TaskPrecedence(jobs[index_job_before - 1], jobs[index_job_after - 1])
# assign resources to jobs
i = 0
for u1, u2 in u:
    if u1 != 0:
        jobs[i].add_required_resource(ps.SelectWorkers(C1_Ress, u1, kind="exact"))
    if u2 != 0:
        jobs[i].add_required_resource(ps.SelectWorkers(C2_Ress, u2, kind="exact"))
    i += 1

# there may be several schedules, add the following constraint
# to find the solution available from the python-MIP documentation
ps.TaskStartAt(jobs[9], 15)
pb.add_objective_makespan()

solver = ps.SchedulingSolver(pb)
solution = solver.solve()

solution.render_gantt_matplotlib()