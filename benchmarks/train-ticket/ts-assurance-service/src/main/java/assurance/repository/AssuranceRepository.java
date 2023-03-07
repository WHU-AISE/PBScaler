package assurance.repository;

import assurance.entity.Assurance;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import java.util.ArrayList;
import java.util.UUID;

/**
 * @author fdse
 */
@Repository
public interface AssuranceRepository  extends MongoRepository<Assurance, String> {

    /**
     * find by id
     *
     * @param id id
     * @return Assurance
     */
    Assurance findById(UUID id);

    /**
     * find by order id
     *
     * @param orderId order id
     * @return Assurance
     */
    @Query("{ 'orderId' : ?0 }")
    Assurance findByOrderId(UUID orderId);

    /**
     * delete by id
     *
     * @param id id
     * @return null
     */
    void deleteById(UUID id);

    /**
     * remove assurance by order id
     *
     * @param orderId order id
     * @return null
     */
    void removeAssuranceByOrderId(UUID orderId);

    /**
     * find all
     *
     * @return ArrayList<Assurance>
     */
    @Override
    ArrayList<Assurance> findAll();
}
