#pragma once

#include <mpi.h>

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Individual/Individual.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Repairs/IRepair.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class MpMoead {
   public:
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariablesNum, int objectivesNum, int neighborhoodSize,
            int migrationInterval, int divisionsNumOfWeightVector, std::shared_ptr<ICrossover<DecisionVariableType>> crossover,
            std::shared_ptr<IDecomposition> decomposition, std::shared_ptr<IMutation<DecisionVariableType>> mutation,
            std::shared_ptr<IProblem<DecisionVariableType>> problem, std::shared_ptr<IRepair<DecisionVariableType>> repair,
            std::shared_ptr<ISampling<DecisionVariableType>> sampling, std::shared_ptr<ISelection> selection)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariablesNum(decisionVariablesNum),
          objectivesNum(objectivesNum),
          neighborhoodSize(neighborhoodSize),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          crossover(crossover),
          decomposition(decomposition),
          mutation(mutation),
          problem(problem),
          repair(repair),
          sampling(sampling),
          selection(selection) {}
    virtual ~MpMoead() {}

    void Run();
    void Initialize();
    void InitializeMpi();
    void InitializeIsland();
    void Update();
    void GetAllObjectives();
    void WriteTransitionOfIdealPoint();

   private:
    int totalPopulationSize;
    int generationNum;
    int decisionVariablesNum;
    int objectivesNum;
    int neighborhoodSize;
    int migrationInterval;
    int divisionsNumOfWeightVector;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<IRepair<DecisionVariableType>> repair;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    int rank;
    int parallelSize;

    std::vector<int> solutionIndexes;
    std::vector<int> externalSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> individuals;
    std::set<int> updatedSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> clonedExternalIndividuals;

    std::vector<Eigen::ArrayXd> transitionOfIdealPoint;

    std::vector<int> GenerateSolutionIndexes();
    std::vector<std::vector<double>> GenerateWeightVectors(int H);
    std::vector<int> GenerateNeighborhoods(std::vector<double>& allWeightVectors);
    std::vector<std::vector<std::pair<double, int>>> CalculateEuclideanDistanceBetweenEachWeightVector(
        std::vector<double>& weightVectors);
    std::vector<int> CalculateNeighborhoodIndexes(std::vector<std::vector<std::pair<double, int>>>& euclideanDistances);

    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                               std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors, std::vector<int>& indexes);
    std::pair<std::vector<int>, std::vector<double>> ScatterExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                                 std::vector<int>& neighborhoodSizes,
                                                                                 std::vector<double>& weightVectors);
    void InitializeIndividualAndWeightVector(std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors);

    void InitializePopulation();
    void InitializeIdealPoint();
    void MakeLocalCopyOfExternalIndividuals();
    std::vector<Individual<DecisionVariableType>> SelectParents(int index);
    Individual<DecisionVariableType> GenerateNewIndividual(int index);
    void UpdateNeighboringIndividuals(int index, Individual<DecisionVariableType>& newIndividual);

    void SendMessage();
    void ReceiveMessage();
    void UpdateWithMessage(std::vector<double>& message);
    std::vector<double> GatherAllObjectives();

#ifdef _TEST_
   public:
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum,
            int migrationInterval, int H)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariablesNum(decisionVariableNum),
          objectivesNum(objectiveNum),
          neighborhoodSize(neighborNum),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(H) {}

    friend class MpMoeadTest;
    friend class MpMoeadTestM;
#endif
};

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Run() {
    Initialize();
    int repeat = generationNum / migrationInterval;
    for (int i = 0; i < repeat; i++) {
        transitionOfIdealPoint.push_back(decomposition->IdealPoint());
        Update();
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeMpi() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Initialize() {
    InitializeMpi();
    InitializeIsland();
    InitializeIdealPoint();
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIsland() {
    std::vector<double> weightVectors1d;
    std::vector<int> neighborhoodIndexes1d;
    if (rank == 0) {
        std::vector<std::vector<double>> weightVectors2d = GenerateWeightVectors(divisionsNumOfWeightVector);
        weightVectors1d = TransformTo1d(weightVectors2d);
        neighborhoodIndexes1d = GenerateNeighborhoods(weightVectors1d);
    }

    std::vector<int> populationSizes;
    if (rank == 0) {
        populationSizes = CalculateNodeWorkloads(totalPopulationSize, parallelSize);
    }

    std::vector<double> receivedWeightVectors = Scatterv(weightVectors1d, populationSizes, objectivesNum, rank, parallelSize);

    std::vector<int> receivedNeighborhoodIndexes =
        Scatterv(neighborhoodIndexes1d, populationSizes, neighborhoodSize, rank, parallelSize);

    std::vector<int> noduplicateNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    std::vector<double> sendExternalNeighboringWeightVectors;
    if (rank == 0) {
        std::tie(noduplicateNeighborhoodIndexes, neighborhoodSizes) =
            GenerateExternalNeighborhood(neighborhoodIndexes1d, populationSizes);
        sendExternalNeighboringWeightVectors = GetWeightVectorsMatchingIndexes(weightVectors1d, noduplicateNeighborhoodIndexes);
    }
    std::vector<double> receivedExternalNeighboringWeightVectors;
    std::tie(externalSolutionIndexes, receivedExternalNeighboringWeightVectors) =
        ScatterExternalNeighborhood(noduplicateNeighborhoodIndexes, neighborhoodSizes, sendExternalNeighboringWeightVectors);

    std::vector<Eigen::ArrayXd> weightVectors = TransformToEigenArrayX2d(receivedWeightVectors, objectivesNum);
    std::vector<std::vector<int>> neighborhoodIndexes = TransformTo2d(receivedNeighborhoodIndexes, neighborhoodSize);
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
        TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors, objectivesNum);

    solutionIndexes = GenerateSolutionIndexes();

    InitializePopulation();

    InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes, externalNeighboringWeightVectors);
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Update() {
    MakeLocalCopyOfExternalIndividuals();
    updatedSolutionIndexes.clear();

    for (int interval = 0; interval < migrationInterval; interval++) {
        for (auto&& i : solutionIndexes) {
            Individual<DecisionVariableType> newIndividual = GenerateNewIndividual(i);
            if (!problem->IsFeasible(newIndividual)) {
                repair->Repair(newIndividual);
            }
            problem->ComputeObjectiveSet(newIndividual);
            decomposition->UpdateIdealPoint(newIndividual.objectives);
            UpdateNeighboringIndividuals(i, newIndividual);
        }
    }

    SendMessage();
    ReceiveMessage();
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GenerateSolutionIndexes() {
    int start = CalculateNodeStartIndex(totalPopulationSize, rank, parallelSize);
    int populationSize = CalculateNodeWorkload(totalPopulationSize, rank, parallelSize);
    std::vector<int> solutionIndexes = Rangei(start, start + populationSize - 1, 1);
    return solutionIndexes;
}

template <typename DecisionVariableType>
std::vector<std::vector<double>> MpMoead<DecisionVariableType>::GenerateWeightVectors(int H) {
    std::vector<double> numeratorOfWeightVector(H + 1);
    std::iota(numeratorOfWeightVector.begin(), numeratorOfWeightVector.end(), 0);
    std::vector<std::vector<double>> product = Product(numeratorOfWeightVector, objectivesNum);
    product.erase(std::remove_if(product.begin(), product.end(), [&](auto v) { return std::reduce(v.begin(), v.end()) != H; }),
                  product.end());
    for (auto&& p : product) {
        for (auto& elem : p) {
            elem /= H;
        }
    }
    return product;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GenerateNeighborhoods(std::vector<double>& allWeightVectors) {
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances =
        CalculateEuclideanDistanceBetweenEachWeightVector(allWeightVectors);
    std::vector<int> neighborhoodIndexes = CalculateNeighborhoodIndexes(euclideanDistances);
    return neighborhoodIndexes;
}

template <typename DecisionVariableType>
std::vector<std::vector<std::pair<double, int>>>
MpMoead<DecisionVariableType>::CalculateEuclideanDistanceBetweenEachWeightVector(std::vector<double>& weightVectors) {
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances(
        totalPopulationSize, std::vector<std::pair<double, int>>(totalPopulationSize));
    for (int i = 0; i < totalPopulationSize; i++) {
        for (int j = 0; j < totalPopulationSize; j++) {
            std::vector<double> diff(objectivesNum);
            for (int k = 0; k < objectivesNum; k++) {
                diff[k] = weightVectors[i * objectivesNum + k] - weightVectors[j * objectivesNum + k];
            }
            double squaredNorm = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            euclideanDistances[i][j] = std::make_pair(squaredNorm, j);
        }
    }
    return euclideanDistances;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::CalculateNeighborhoodIndexes(
    std::vector<std::vector<std::pair<double, int>>>& euclideanDistances) {
    std::vector<int> neighborhoodIndexes(totalPopulationSize * neighborhoodSize);
    for (std::size_t i = 0; i < totalPopulationSize; i++) {
        std::sort(euclideanDistances[i].begin(), euclideanDistances[i].end());
        for (std::size_t j = 0; j < neighborhoodSize; j++) {
            neighborhoodIndexes[i * neighborhoodSize + j] = euclideanDistances[i][j].second;
        }
    }
    return neighborhoodIndexes;
}

template <typename DecisionVariableType>
std::pair<std::vector<int>, std::vector<int>> MpMoead<DecisionVariableType>::GenerateExternalNeighborhood(
    std::vector<int>& neighborhoodIndexes, std::vector<int>& populationSizes) {
    std::vector<int> noduplicateNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    for (int i = 0; i < populationSizes.size(); i++) {
        int start = std::reduce(populationSizes.begin(), populationSizes.begin() + i);
        int end = start + populationSizes[i];

        std::vector<int> indexes(neighborhoodIndexes.begin() + (start * neighborhoodSize),
                                 neighborhoodIndexes.begin() + (end * neighborhoodSize));

        // 重複を削除
        std::sort(indexes.begin(), indexes.end());
        indexes.erase(std::unique(indexes.begin(), indexes.end()), indexes.end());

        // 自分の担当する解のインデックスを削除
        std::erase_if(indexes, [&](int index) { return start <= index && index < end; });

        noduplicateNeighborhoodIndexes.insert(noduplicateNeighborhoodIndexes.end(), indexes.begin(), indexes.end());
        neighborhoodSizes.push_back(indexes.size());
    }
    return {noduplicateNeighborhoodIndexes, neighborhoodSizes};
}

template <typename DecisionVariableType>
std::vector<double> MpMoead<DecisionVariableType>::GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors,
                                                                                   std::vector<int>& indexes) {
    std::vector<double> matchingWeightVectors;
    for (int i = 0; i < indexes.size(); i++) {
        matchingWeightVectors.insert(matchingWeightVectors.end(), weightVectors.begin() + indexes[i] * objectivesNum,
                                     weightVectors.begin() + (indexes[i] + 1) * objectivesNum);
    }
    return matchingWeightVectors;
}

template <typename DecisionVariableType>
std::pair<std::vector<int>, std::vector<double>> MpMoead<DecisionVariableType>::ScatterExternalNeighborhood(
    std::vector<int>& neighborhoodIndexes, std::vector<int>& neighborhoodSizes, std::vector<double>& weightVectors) {
    std::vector<int> receivedNeighborhoodIndexes;
    receivedNeighborhoodIndexes = Scatterv(neighborhoodIndexes, neighborhoodSizes, 1, rank, parallelSize);

    std::vector<double> receivedWeightVectors;
    receivedWeightVectors = Scatterv(weightVectors, neighborhoodSizes, objectivesNum, rank, parallelSize);

    return {receivedNeighborhoodIndexes, receivedWeightVectors};
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializePopulation() {
    int sampleNum = solutionIndexes.size() + externalSolutionIndexes.size();
    std::vector<Individual<DecisionVariableType>> sampledIndividuals = sampling->Sample(sampleNum, decisionVariablesNum);

    int sampleIndex = 0;
    for (int i = 0; i < solutionIndexes.size(); i++, sampleIndex++) {
        individuals[solutionIndexes[i]] = sampledIndividuals[sampleIndex];
        problem->ComputeObjectiveSet(individuals[solutionIndexes[i]]);
    }
    for (int i = 0; i < externalSolutionIndexes.size(); i++, sampleIndex++) {
        clonedExternalIndividuals[externalSolutionIndexes[i]] = sampledIndividuals[sampleIndex];
        problem->ComputeObjectiveSet(clonedExternalIndividuals[externalSolutionIndexes[i]]);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIndividualAndWeightVector(
    std::vector<Eigen::ArrayXd>& weightVectors, std::vector<std::vector<int>>& neighborhoodIndexes,
    std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
    for (int i = 0; i < solutionIndexes.size(); i++) {
        individuals[solutionIndexes[i]].weightVector = weightVectors[i];
        individuals[solutionIndexes[i]].neighborhood = neighborhoodIndexes[i];
    }
    for (int i = 0; i < externalSolutionIndexes.size(); i++) {
        clonedExternalIndividuals[externalSolutionIndexes[i]].weightVector = externalNeighboringWeightVectors[i];
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIdealPoint() {
    for (auto&& individual : individuals) {
        decomposition->UpdateIdealPoint(individual.second.objectives);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::MakeLocalCopyOfExternalIndividuals() {
    for (auto&& i : externalSolutionIndexes) {
        individuals[i] = clonedExternalIndividuals[i];
    }
}

template <typename DecisionVariableType>
std::vector<Individual<DecisionVariableType>> MpMoead<DecisionVariableType>::SelectParents(int index) {
    std::vector<int> parentCandidates;
    std::copy_if(individuals[index].neighborhood.begin(), individuals[index].neighborhood.end(),
                 std::back_inserter(parentCandidates), [index](int i) { return i != index; });
    std::vector<int> parentIndexes = selection->Select(crossover->GetParentNum() - 1, parentCandidates);
    std::vector<Individual<DecisionVariableType>> parents;
    parents.push_back(individuals[index]);
    for (auto&& i : parentIndexes) {
        parents.push_back(individuals[i]);
    }
    return parents;
}

template <typename DecisionVariableType>
Individual<DecisionVariableType> MpMoead<DecisionVariableType>::GenerateNewIndividual(int index) {
    std::vector<Individual<DecisionVariableType>> parents = SelectParents(index);
    Individual<DecisionVariableType> newIndividual = crossover->Cross(parents);
    mutation->Mutate(newIndividual);
    return newIndividual;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::UpdateNeighboringIndividuals(int index, Individual<DecisionVariableType>& newIndividual) {
    for (auto&& i : individuals[index].neighborhood) {
        double newSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, newIndividual.objectives);
        double oldSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, individuals[i].objectives);
        if (newSubObjective < oldSubObjective) {
            individuals[i].UpdateFrom(newIndividual);
            if (i == index) {
                updatedSolutionIndexes.insert(i);
            }
        }
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::SendMessage() {
    std::unordered_map<int, std::pair<std::set<int>, std::vector<double>>> send;
    for (auto&& i : externalSolutionIndexes) {
        int process = GetRankFromIndex(totalPopulationSize, i, parallelSize);
        send[process] = std::make_pair(std::set<int>(), std::vector<double>());
    }
    for (auto&& i : externalSolutionIndexes) {
        int process = GetRankFromIndex(totalPopulationSize, i, parallelSize);

        bool isUpdated = (individuals[i].solution != individuals[i].solution).all();
        if (isUpdated) {
            send[process].first.insert(i);
            send[process].second.push_back(i);
            send[process].second.insert(send[process].second.end(), individuals[i].solution.begin(),
                                        individuals[i].solution.end());
        }

        for (auto&& j : updatedSolutionIndexes) {
            bool contains = std::find(individuals[j].neighborhood.begin(), individuals[j].neighborhood.end(), i) !=
                            individuals[j].neighborhood.end();
            if (!contains) {
                continue;
            }
            bool alreadyAdded = send[process].first.find(j) != send[process].first.end();
            if (alreadyAdded) {
                continue;
            }
            send[process].first.insert(j);
            send[process].second.push_back(j);
            send[process].second.insert(send[process].second.end(), individuals[j].solution.begin(),
                                        individuals[j].solution.end());
        }
    }

    for (auto&& [process, value] : send) {
        if (value.second.empty()) {
            continue;
        }

        int sendDataSize = value.second.size();
        MPI_Request request;
        MPI_Isend(&sendDataSize, 1, MPI_INT, process, 0, MPI_COMM_WORLD, &request);
        MPI_Isend(value.second.data(), sendDataSize, MPI_DOUBLE, process, 1, MPI_COMM_WORLD, &request);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::ReceiveMessage() {
    for (auto&& i : externalSolutionIndexes) {
        int isReceived;
        int source = GetRankFromIndex(totalPopulationSize, i, parallelSize);
        MPI_Iprobe(source, 0, MPI_COMM_WORLD, &isReceived, MPI_STATUS_IGNORE);

        if (!isReceived) {
            continue;
        }

        int receiveDataSize;
        MPI_Recv(&receiveDataSize, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<double> receiveData(receiveDataSize);
        MPI_Recv(receiveData.data(), receiveDataSize, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        UpdateWithMessage(receiveData);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::UpdateWithMessage(std::vector<double>& message) {
    for (int j = 0; j < message.size(); j += decisionVariablesNum + 1) {
        int index = message[j];
        Eigen::ArrayX<DecisionVariableType> newSolution =
            Eigen::Map<Eigen::ArrayX<DecisionVariableType>>(message.data() + j + 1, decisionVariablesNum);
        Individual<DecisionVariableType> newIndividual(newSolution);
        problem->ComputeObjectiveSet(newIndividual);
        bool isExternal =
            std::find(externalSolutionIndexes.begin(), externalSolutionIndexes.end(), index) != externalSolutionIndexes.end();
        if (isExternal) {
            clonedExternalIndividuals[index].UpdateFrom(newIndividual);
        } else {
            double newSubObjective = decomposition->ComputeObjective(individuals[index].weightVector, newIndividual.objectives);
            double oldSubObjective =
                decomposition->ComputeObjective(individuals[index].weightVector, individuals[index].objectives);
            if (newSubObjective < oldSubObjective) {
                individuals[index].UpdateFrom(newIndividual);
                updatedSolutionIndexes.insert(index);
            }
        }

        decomposition->UpdateIdealPoint(newIndividual.objectives);
    }
}

template <typename DecisionVariableType>
std::vector<double> MpMoead<DecisionVariableType>::GatherAllObjectives() {
    std::vector<double> sendObjectives;
    for (auto&& i : solutionIndexes) {
        sendObjectives.insert(sendObjectives.end(), individuals[i].objectives.begin(), individuals[i].objectives.end());
    }

    int dataSize = sendObjectives.size();
    std::vector<int> dataCounts;
    if (rank == 0) {
        dataCounts.resize(parallelSize);
    }
    MPI_Gather(&dataSize, 1, MPI_INT, dataCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int totalDataSize = 0;
    std::vector<int> displacements;
    if (rank == 0) {
        for (int i = 0; i < dataCounts.size(); i++) {
            displacements.push_back(totalDataSize);
            totalDataSize += dataCounts[i];
        }
    }

    std::vector<double> receiveObjectives;
    if (rank == 0) {
        receiveObjectives.resize(totalDataSize);
    }
    MPI_Gatherv(sendObjectives.data(), dataSize, MPI_DOUBLE, receiveObjectives.data(), dataCounts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return receiveObjectives;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::GetAllObjectives() {
    if (rank == 0) {
        std::filesystem::create_directories("out/data");
        std::filesystem::create_directories("out/data/mp_moead/");
        std::filesystem::create_directories("out/data/mp_moead/objective");

        for (const auto& entry : std::filesystem::directory_iterator("out/data/mp_moead/objective")) {
            std::filesystem::remove(entry.path());
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::string filename = "out/data/mp_moead/objective/objective-" + std::to_string(rank) + ".txt";
    std::ofstream ofs(filename);
    for (auto&& individual : individuals) {
        for (int i = 0; i < individual.second.objectives.size(); i++) {
            ofs << individual.second.objectives(i);
            if (i != individual.second.objectives.size() - 1) {
                ofs << " ";
            }
        }
        ofs << std::endl;
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::WriteTransitionOfIdealPoint() {
    if (rank == 0) {
        std::filesystem::create_directories("out/data");
        std::filesystem::create_directories("out/data/mp_moead/");
        std::filesystem::create_directories("out/data/mp_moead/ideal_point");

        for (const auto& entry : std::filesystem::directory_iterator("out/data/mp_moead/ideal_point")) {
            std::filesystem::remove(entry.path());
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::string filename = "out/data/mp_moead/ideal_point/ideal_point-" + std::to_string(rank) + ".txt";
    std::ofstream ofs(filename);
    for (auto&& idealPoint : transitionOfIdealPoint) {
        for (int i = 0; i < idealPoint.size(); i++) {
            ofs << idealPoint[i];
            if (i != idealPoint.size() - 1) {
                ofs << " ";
            }
        }
        ofs << std::endl;
    }
}

}  // namespace Eacpp
