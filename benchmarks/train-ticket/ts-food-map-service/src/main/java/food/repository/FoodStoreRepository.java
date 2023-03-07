package food.repository;

import food.entity.FoodStore;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface FoodStoreRepository extends MongoRepository<FoodStore, String> {

    FoodStore findById(UUID id);

    @Query("{ 'stationId' : ?0 }")
    List<FoodStore> findByStationId(String stationId);
    List<FoodStore> findByStationIdIn(List<String> stationIds);


    @Override
    List<FoodStore> findAll();

    void deleteById(UUID id);
}
