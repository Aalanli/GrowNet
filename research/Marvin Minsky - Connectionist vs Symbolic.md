> Thus, contemporary symbolic AI systems are now too constrained to be able to deal with exceptions to rules, or to exploit fuzzy, approximate, or heuristic fragments of knowledge. Partly in reaction to this, the connectionist movement initially tried to develop more flexible systems, but soon came to be imprisoned in its own peculiar ideology---of trying to build learning systems endowed with as little architectural structure as possible, hoping to create machines that could serve all masters equally well. The trouble with this is that even a seemingly neutral architecture still embodies an implicit assumption about which things are presumed to be "similar."

> all AI researchers seek to make machines that solve problems

> Thus, the present-day systems of both types show serious limitations. The top-down systems are handicapped by inflexible mechanisms for retrieving knowledge and reasoning about it, while the bottom-up systems are crippled by inflexible architectures and organizational schemes. Neither type of system has been developed so as to be able to exploit multiple, diverse varieties of knowledge.

> The knowledge must be embodied in some form of mechanism, data-structure, or other representation

- There seems to be an issue with (first order) logic in its inability to express connections between concepts, and its apparent strictness, which renderes optimization difficult. 
- There is no obvious distance metric between 'representations' in logic. Eg. $A\lor B\lor C$, $(AB) \lor (AC) \lor (BC)$
- Universal quantifiers and logical predicates are defined too narrowly to be useful


> There are several points in SOM that suggest that commonsense reasoning systems may not need to increase in the density of physical connectivity as fast as they increase the complexity and scope of their performances. 
> Chapter 6 argues that knowledge systems must evolve into clumps of specialized agencies, rather than homogeneous networks, because they develop different types of internal representations. When this happens, it will become neither feasible nor practical for any of those agencies to communicate directly with the interior of others.

> If our minds are assembled of agencies with so little inter-communication, how can those parts cooperate?

> All this suggests (but does not prove) that large commonsense reasoning systems will not need to be "fully connected." Instead, the system could consist of localized clumps of expertise.

> Eventually, we should be able to build a sound technical theory about the connection densities required for commonsense thinking, but I don't think that we have the right foundations as yet. The problem is that contemporary theories of computational complexity are still based too much on worst-case analyses, or on coarse statistical assumptions---neither of which suitably represents realistic heuristic conditions. The worst-case theories unduly emphasize the intractable versions of problems which, in their usual forms, present less practical difficulty. The statistical theories tend to uniformly weight all instances, for lack of systematic ways to emphasize the types of situations of most practical interest. But the AI systems of the future, like their human counterparts, will normally prefer to satisfy rather than optimize---and we don't yet have theories that can realistically portray those mundane sorts of requirements.