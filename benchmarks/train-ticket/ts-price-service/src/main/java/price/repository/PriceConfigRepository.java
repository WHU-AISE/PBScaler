package price.repository;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import price.entity.PriceConfig;
import java.util.List;
import java.util.UUID;

/**
 * @author fdse
 */
@Repository
public interface PriceConfigRepository extends MongoRepository<PriceConfig, String> {

    @Query("{ 'id': ?0 }")
    PriceConfig findById(UUID id);

    @Query("{ 'routeId': ?0 , 'trainType': ?1 }")
    PriceConfig findByRouteIdAndTrainType(String routeId,String trainType);

    @Override
    List<PriceConfig> findAll();

}
