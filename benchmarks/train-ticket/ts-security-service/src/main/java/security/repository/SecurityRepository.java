package security.repository;

import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import security.entity.SecurityConfig;
import java.util.ArrayList;
import java.util.UUID;

/**
 * @author fdse
 */
@Repository
public interface SecurityRepository extends MongoRepository<SecurityConfig,String>{

    @Query("{ 'name': ?0 }")
    SecurityConfig findByName(String name);

    @Query("{ 'id': ?0 }")
    SecurityConfig findById(UUID id);

    @Override
    ArrayList<SecurityConfig> findAll();

    void deleteById(UUID id);
}
