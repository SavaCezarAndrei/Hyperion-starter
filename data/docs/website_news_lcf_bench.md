# LCF-Bench: A Reproducible Benchmark for Feasibility-Aware Logical Cluster Formation in the Cloud–Edge-IoT Computing Continuum

                    31, July, 2026

                                            ·5 minutes
 read

                                                                ·

- Publication banner

            HYPER-AI partners have published LCF-Bench, an open benchmark that tests how well different strategies pick a group of machines to host an application across cloud, edge and IoT infrastructure. The results show that success does not fade away gradually as demand grows. It falls off a cliff.
The problem is harder than it looks
Nowadays, modern applications no longer run in one place but they are rather spread across large cloud data centres, smaller servers at the edge of the network, and tiny devices out in the field. Before such an application can start, someone has to decide which machines will host it.
At first glance this seems like simple arithmetic. Add up what the application needs, add up what the machines can offer, and if the second number is bigger than the first, you are done.
Unfortunately it does not work that way. In this publication, a simple example is given: imagine three machines with four processor cores each, so twelve cores in total. Now imagine four tasks that each need three cores, so twelve cores in total again. The capacity matches the demand exactly, and every single task fits comfortably on any of the machines. And yet the job cannot be done, because once you place one task on each machine, the fourth task has nowhere to go. Only one core is left free on each machine and the task needs three of them together.
This is the heart of the matter. A task cannot be cut into pieces and spread around. It needs its processor power, its memory and its storage all on the same machine at the same time. So the real question is not "is there enough in total?" but "is there an arrangement that actually works?"
Why a new benchmark was needed
Plenty of tools already exist for studying cloud and edge systems, and Kubernetes has become the standard way to run containers across clusters. But according to the study here, none of them isolates this particular question. They all blend the choice of machines together with the placement of individual tasks, resolving both at the same time and one task at a time.
That makes it very difficult to study the choice of machines on its own, or to compare one selection strategy against another under identical conditions. There was, in short, no common yardstick.
LCF-Bench is designed to be that yardstick. It comes with three main parts: a generator that produces test cases with controlled sizes and controlled pressure, an export function that turns every test case into something the real Kubernetes scheduler can chew on, and a standard scoring procedure so that every method is judged in exactly the same way. Anyone can plug in their own method and compare it against the reference results.
Five approaches were used to calibrate the benchmark. Round-Robin simply cycles through the machines in order and serves as a deliberate worst case. First-Fit puts each task on the first machine with room. Best-Fit puts each task where it leaves the least wasted space. A mathematical optimiser gives the theoretically best possible answer. And the default Kubernetes scheduler stands in for what happens in a real production system today.
What the tests revealed
Feasibility falls off a cliff. As the researchers slowly increased how much each task demanded, every method kept working perfectly, and then suddenly stopped working almost entirely. There is no gentle decline to warn you that trouble is coming. This means operators cannot rely on watching performance degrade as an early warning sign.
Smarter packing buys real headroom. The cliffs did not all arrive at the same moment, and the order was consistent. Round-Robin and First-Fit gave way first, Best-Fit held on considerably longer, and the mathematical optimiser held on longest of all. In the reference runs Best-Fit stayed fully successful even when demand was around eighty percent above the nominal level, which is a useful safety margin to have in practice.
The perfect answer is too slow to use. The mathematical optimiser produced the best clusters every time, wasting only a few percent of capacity compared to what was strictly needed. But it hit its twenty second time limit on anything beyond small problems and had to be skipped entirely at larger sizes. It is a valuable reference point, not something you would run in production.
Kubernetes today has less margin than expected. The default scheduler always found an answer when resources were plentiful, but under pressure it gave up earlier than Best Fit did. It also became dramatically slower as things got tight, taking over a minute in the hardest cases because it keeps retrying individual placements that cannot succeed.
The type of workload changes everything. When demand was dominated by storage rather than processing, the picture shifted noticeably. The gap between clever and simple methods narrowed, and the mathematical optimiser suddenly became easy to run rather than impossibly slow. The practical lesson is that there is no single right answer. The best strategy depends on which resource is the bottleneck for your particular application.
Simple scoring can waste resources in surprising ways. Because the heuristic methods judge each candidate machine using one combined score, they occasionally accept a machine with enormous storage capacity simply because it looks good overall. The result is a cluster that is efficient on processing and memory but sits on far more storage than it will ever use.
Why it matters
For anyone building orchestration systems across the cloud to edge continuum, the message is practical. Choose your machines with a method that thinks about how tasks will actually fit together, keep a healthy margin above your expected demand, and tune your choice to the kind of workload you expect to run. And because the cliff is sharp, do not assume you will get advance warning.
The benchmark also gives the research community something it did not have before, which is a shared and repeatable way to test new ideas. Anyone proposing a better method can now run it on exactly the same test cases as everyone else. 
Read the Publication Here
