package foodsearch.repository;

import foodsearch.entity.FoodOrder;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface FoodOrderRepository extends MongoRepository<FoodOrder, String> {

    FoodOrder findById(UUID id);

    @Query("{ 'orderId' : ?0 }")
    FoodOrder findByOrderId(UUID orderId);

    @Override
    List<FoodOrder> findAll();

    void deleteById(UUID id);

    void deleteFoodOrderByOrderId(UUID id);

}
